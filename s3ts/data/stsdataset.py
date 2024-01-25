import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule

from s3ts.data.base import STSDataset
from s3ts.data.methods import reduce_imbalance

from s3ts.api.gaf_mtf import mtf_compute, gaf_compute

class StreamingTimeSeries(STSDataset):

    def __init__(self,
            STS: np.ndarray,
            SCS: np.ndarray,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        self.STS = STS
        self.SCS = SCS

        self.splits = np.array([0, SCS.shape[0]])

        # process ds
        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")


class StreamingTimeSeriesCopy(Dataset):

    def __init__(self,
            stsds: StreamingTimeSeries, indices: np.ndarray, label_mode: int = 1, mode: str = None, mtf_bins: int = 30
            ) -> None:
        super().__init__()

        assert label_mode%2==1

        self.stsds = stsds
        self.indices = indices
        self.label_mode = label_mode
        self.mode = mode
        self.mtf_bins = mtf_bins
        
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:

        ts, c = self.stsds[self.indices[index]]

        if self.label_mode > 1:
            c = torch.mode(c[-self.label_mode:]).values
        else:
            c = c[-1]

        
        if self.mode == "gasf":
            transformed = gaf_compute(ts, "s", (-1, 1))
            return {"series": ts, "label": c, "transformed": transformed}

        elif self.mode == "gadf":
            transformed = gaf_compute(ts, "d", (-1, 1))
            return {"series": ts, "label": c, "transformed": transformed}

        elif self.mode == "mtf":
            transformed = mtf_compute(ts, self.mtf_bins, (-1, 1))
            return {"series": ts, "label": c, "transformed": transformed}
        
        elif self.mode == "fft":
            transformed = torch.fft.fft(ts, dim=-1)
            transformed = torch.cat([transformed.real, transformed.imag], dim=0)
            return {"series": ts, "label": c, "transformed": transformed}

        else:
            return {"series": ts, "label": c}
    
    def __del__(self):
        del self.stsds


class LSTSDataset(LightningDataModule):

    """ Data module for the experiments. """

    STS: np.ndarray     # data stream
    SCS: np.ndarray     # class stream
    DM: np.ndarray      # dissimilarity matrix

    data_split: dict[str: np.ndarray]    
                        # train / val / test split
    batch_size: int     # dataloader batch size

    def __init__(self,
            stsds: STSDataset,    
            data_split: dict, batch_size: int, 
            random_seed: int = 42, 
            num_workers: int = 1,
            reduce_train_imbalance: bool = False,
            label_mode: int = 1,
            overlap: int = -1,
            mode: str = None,
            mtf_bins: int = 50
            ) -> None:

        # save parameters as attributes
        super().__init__()
        
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers

        self.stsds = stsds
        self.wdw_len = self.stsds.wsize
        self.wdw_str = self.stsds.wstride
        self.sts_str = False

        self.l_patterns = None

        # gather dataset info   
        self.n_dims = self.stsds.STS.shape[0]
        self.n_classes = len(np.unique(self.stsds.SCS))
        self.n_patterns = self.n_classes

        # convert to tensors
        if not torch.is_tensor(self.stsds.STS):
            self.stsds.STS = torch.from_numpy(self.stsds.STS).to(torch.float32)
        if not torch.is_tensor(self.stsds.SCS):
            self.stsds.SCS = torch.from_numpy(self.stsds.SCS).to(torch.int64)

        skip = 1 if overlap == -1 else self.wdw_len - overlap
        if skip < 1:
            raise Exception(f"Overlap must be smaller than window size, overlap:{overlap}, window_size {self.wdw_len}")

        total_observations = self.stsds.indices.shape[0]
        train_indices = np.arange(total_observations)[data_split["train"](self.stsds.indices)][::skip]
        test_indices = np.arange(total_observations)[data_split["test"](self.stsds.indices)][::skip]
        val_indices = np.arange(total_observations)[data_split["val"](self.stsds.indices)][::skip]

        self.train_labels = self.stsds.SCS[self.stsds.indices[train_indices]]
        self.train_label_weights = np.empty_like(self.train_labels, dtype=np.float32)

        self.reduce_train_imbalance = reduce_train_imbalance
        if reduce_train_imbalance:
            cl, counts = torch.unique(self.train_labels, return_counts=True)
            for i in range(cl.shape[0]):
                self.train_label_weights[self.train_labels == cl[i]] = self.train_labels.shape[0] / counts[i]

            examples_per_epoch = int(counts.float().mean().ceil().item())
            print(f"Sampling {examples_per_epoch} (balanced) observations per epoch.")
            self.train_sampler = WeightedRandomSampler(self.train_label_weights, int(counts.float().mean().ceil().item()), replacement=True)
            # train_indices = reduce_imbalance(train_indices, self.train_labels, seed=random_seed)

        self.ds_train = StreamingTimeSeriesCopy(self.stsds, train_indices, label_mode, mode, mtf_bins)
        self.ds_test = StreamingTimeSeriesCopy(self.stsds, test_indices, label_mode, mode, mtf_bins)
        self.ds_val = StreamingTimeSeriesCopy(self.stsds, val_indices, label_mode, mode, mtf_bins)
        
    def train_dataloader(self) -> DataLoader:
        """ Returns the training DataLoader. """
        if self.reduce_train_imbalance:
            return DataLoader(self.ds_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, sampler=self.train_sampler,
                pin_memory=True, persistent_workers=True)
        else:
            return DataLoader(self.ds_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True ,
                pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_val, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)
    
    def predict_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)