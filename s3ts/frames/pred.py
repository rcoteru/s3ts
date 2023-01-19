# external imports
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv
import torch

import multiprocessing as mp
import numpy as np

# ================================================================= #

class PredDataset(Dataset):

    def __init__(self,
            frames: np.ndarray,
            series: np.ndarray,
            labels: np.ndarray,
            indexes: np.ndarray,
            lab_shifts: list[int],
            window_size: int,
            transform = None, 
            target_transform = None
            ) -> None:

        self.frames = frames
        self.series = series
        self.labels = labels
        self.indexes = indexes

        self.n_samples = len(self.indexes)
        self.window_size = window_size
        self.lab_shifts = lab_shifts

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """ Return the number of samples in the dataset. """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (x, y) from the dataset. """

        idx = self.indexes[idx]
        mask = np.zeros(len(self.series))
        for s in self.lab_shifts:
            mask[idx+s] = 1

        label = self.labels[mask]
        frame = self.frames[:,:,idx - self.window_size:idx]
        
        if self.transform:
            frame = self.transform(frame)
        if self.target_transform:
            label = self.target_transform(label)

        return frame, label

# ================================================================= #

class PredDataModule(LightningDataModule):

    def __init__(self,
            # calculate this outside
            STS: np.ndarray,
            DFS: np.ndarray,
            labels: np.ndarray,
            # ~~~~~~~~~~~~~~~~~~~~~~
            window_size: int, 
            batch_size: int,
            test_size: float,
            lab_shifts: list[int],
            random_state: int = 0,
            eval_size: float = 0.10,
            ) -> None:
        
        super().__init__()

        # convert dataset to tensors
        self.STS = torch.from_numpy(STS).to(torch.float32)
        self.DFS = torch.from_numpy(DFS).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(torch.int64)
        self.labels = torch.nn.functional.one_hot(self.labels)

        # get dataset info
        self.n_labels = len(np.unique(labels))
        self.n_patterns = DFS.shape[0]
        self.l_patterns = DFS.shape[1]
        self.l_DFS = DFS.shape[2]

        # datamodule settings
        self.batch_size = batch_size
        self.window_size = window_size
        self.lab_shifts = np.array(lab_shifts, dtype=int)
        self.random_state = random_state 
        self.num_workers = mp.cpu_count()//2

        # generate train/eval/test indexes
        br0 = self.window_size*3
        br1 = int(self.l_DFS*(1-test_size-eval_size)) + max(window_size, np.max(lab_shifts))
        br2 = int(self.l_DFS*(1-test_size)) + max(window_size, np.max(lab_shifts))
        end = self.l_DFS - max(window_size, np.max(lab_shifts))

        self.train_idx = np.arange(br0, br1)
        self.eval_idx = np.arange(br1, br2)
        self.test_idx = np.arange(br2, end)

        print("Train samples:", len(self.train_idx))
        print("Eval samples:", len(self.eval_idx))
        print("Test samples:", len(self.test_idx))

        # normalization_transform
        transform = tv.transforms.Normalize(
                self.DFS[:,:,br0:br1].mean(axis=[1,2]),
                self.DFS[:,:,br0:br1].std(axis=[1,2]))

        # training dataset
        self.ds_train = PredDataset(indexes=self.train_idx, lab_shifts=lab_shifts,
            frames=self.DFS, series=self.STS, labels=self.labels, window_size=self.window_size, transform=transform)

        self.ds_eval = PredDataset(indexes=self.eval_idx, lab_shifts=lab_shifts,
            frames=self.DFS, series=self.STS, labels=self.labels, window_size=self.window_size, transform=transform)

        self.ds_test = PredDataset(indexes=self.test_idx, lab_shifts=lab_shifts,
            frames=self.DFS, series=self.STS, labels=self.labels, window_size=self.window_size, transform=transform)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def train_dataloader(self):
        """ Returns the training DataLoader. """
        return DataLoader(self.ds_train, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def val_dataloader(self):
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_eval, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def test_dataloader(self):
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def predict_dataloader(self):
        """ Returns the pred DataLoader. (test) """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False)