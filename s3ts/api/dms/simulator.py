#/usr/bin/env python3
# -*- coding: utf-8 -*-

from pandas import Series
from s3ts.api.simulation import StreamSimulator, samples_from_simulator
from s3ts.api.ts2sts import fin_random_STS, fin_balanced_STS
from s3ts.api.encodings import compute_DM, compute_GM
from s3ts.api.dms.base import StreamingFramesDM

from torch.utils.data import Dataset, DataLoader
from typing import Optional, Literal, Any
import multiprocessing as mp

import torchvision as tv
import numpy as np
import torch

class SampleDS(Dataset):

    series: torch.Tensor
    labels: torch.Tensor
    frames: torch.Tensor
    img_trans: torch.nn.Module

    def __init__(self, series: torch.Tensor, labels: torch.Tensor, frames: torch.Tensor,
        img_trans: Optional[torch.nn.Module] = None) -> None:
        self.__dict__.update(locals())

    def __len__(self) -> int:
        return self.series.shape[0]
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame = self.frames[idx]
        series = self.series[idx]
        label = self.labels[idx]
        if self.img_trans:
            frame = self.img_trans(frame)
        return {"frame": frame, "series": series, "label": label}
    
class ContinuousDS(Dataset):

    DM: torch.Tensor
    STS: torch.Tensor
    SCS: torch.Tensor

    wdw_len: int
    wdw_str: int
    sts_str: bool
    margin: int

    DM_trans: torch.nn.Module
    STS_trans: torch.nn.Module

    def __init__(self, DM, STS, SCS, wdw_len, wdw_str, sts_str, 
            margin, DM_trans=None, STS_trans=None) -> None:
        self.__dict__.update(locals())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        # index
        idx = self.index[idx]
        # frame
        frame = self.DM[:,:,idx - self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]
        if self.DM_trans:
            frame = self.DM_trans(frame)
        # series
        if self.sts_str:
            series = self.STS[:,idx-self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]
        else:
            series = self.STS[:,idx-self.wdw_len*self.wdw_str+1:idx+1]
        if self.STS_trans:
            series = self.STS_trans(series)
        # label
        label = self.SCS[idx]

        return {"frame": frame, "series": series, "label": label}

class DynamicDM(StreamingFramesDM):

    """ Data module for the experiments. """

    X: np.ndarray      # data stream
    Y: np.ndarray      # class stream

    series: dict[Literal["train", "val", "test"], torch.Tensor]
    labels: dict[Literal["train", "val", "test"], torch.Tensor]
    frames: dict[Literal["train", "val", "test"], torch.Tensor]      # dissimilarity matrix

    sts_method: Literal["random", "balanced"]
    image_method: Literal["df", "gf", None]
    image_args: dict[str, Any]

    # data split info
    data_split: dict[Literal["train", "val", "test"], np.ndarray]       # train / val / test split
    nsamps_train: dict[str, int]            # train / val number of samples
    nevents_test: int                       # number of test events
    tsamp_args: dict[str, Any]
    
    batch_size: int     # dataloader batch size
    
    seed: int           # random seed
    num_workers: int    # dataloader nworkers

    def __init__(self, X: np.ndarray, Y: np.ndarray, patts: np.ndarray,
            sts_method: Literal["random", "balanced"], img_method: Literal["df", "gf", None],    
            wdw_len: int, wdw_str: int, sts_str: bool,
            batch_size: int, data_split: dict[str, np.ndarray], 
            nsamps_train: dict[str, int], nevents_test: int, 
            random_seed: int = 42, 
            tsamp_args: dict[str, Any] = {},
            image_args: dict[str, Any] = {},
            num_workers: int = mp.cpu_count()//2,
            ) -> None:
        
        # save parameters as attributes
        super().__init__()
        self.__dict__.update(locals())

        # gather dataset info
        self.n_dims = X.shape[1]
        self.n_classes = len(np.unique(Y))
        self.n_patterns = patts.shape[0]
        self.l_patterns = patts.shape[2]

        # get idx for each set
        train_idx = self.data_split["train"]
        val_idx = self.data_split["val"]
        test_idx = self.data_split["test"]

        # create train / val samples from simulators
        margin = self.wdw_len*self.wdw_str+1
        self.sims: dict[Literal["train", "val"], StreamSimulator] = {}
        self.series: dict[Literal["train", "val", "test"], torch.Tensor] = {}
        self.labels: dict[Literal["train", "val", "test"], torch.Tensor] = {}
        self.frames: dict[Literal["train", "val", "test"], torch.Tensor] = {}
        for split in ["train", "val"]:
            # create simulator
            self.sims[split] = StreamSimulator( # type: ignore
                X=X[data_split[split]], Y=Y[data_split[split]], patts=patts,
                wdw_len=self.wdw_len, wdw_str=self.wdw_str,
                sts_method=self.sts_method, image_method=self.image_method,
                image_args=self.image_args, random_seed=self.seed,
                discard=margin)
            # create samples
            series, labels, frames = samples_from_simulator(sim=self.sims[split],  # type: ignore
                    nsamps=self.nsamps_train[split], **self.tsamp_args)
            # convert to tensors
            self.series[split] = torch.from_numpy(series).to(torch.float32) # type: ignore
            self.labels[split] = torch.from_numpy(labels).to(torch.int64) # type: ignore
            self.frames[split] = torch.from_numpy(frames).to(torch.float32) # type: ignore

        # create test STS / SCS
        if sts_method == "random":
            tSTS, tSCS = fin_random_STS(X=X[test_idx], Y=Y[test_idx], length=nevents_test, seed=self.seed)
            self.series["test"] = torch.from_numpy(tSTS).to(torch.float32)
            self.labels["test"] = torch.from_numpy(tSCS).to(torch.int64)
            
        elif sts_method == "balanced":
            tSTS, tSCS = fin_balanced_STS(X=X[test_idx], Y=Y[test_idx], length=nevents_test, seed=self.seed)
            self.series["test"] = torch.from_numpy(tSTS).to(torch.float32)
            self.labels["test"] = torch.from_numpy(tSCS).to(torch.int64)

        # create test DM
        tDM = np.zeros((self.n_patterns, self.l_patterns, nevents_test*self.X.shape[2]))
        if self.image_method is not None:
            if self.image_method == "df":
                tDM = compute_DM(STS=tSTS, patts=patts, **self.image_args)
            if self.image_method == "gf":
                tDM = compute_GM(STS=tSTS, patts=patts, **self.image_args)
        self.frames["test"] = torch.from_numpy(tDM).to(torch.float32)

        # create transforms
        if img_method is not None:
            img_trans = tv.transforms.Normalize(
                self.frames["train"].mean(dim=[1,2]),
                self.frames["train"].std(dim=[1,2]))
        else:
            img_trans = None

        # create datasets
        self.ds_train = SampleDS(series=self.series["train"], labels=self.labels["train"],
            frames=self.frames["train"], img_trans=img_trans)
        self.ds_val = SampleDS(series=self.series["val"], labels=self.labels["val"],
            frames=self.frames["val"], img_trans=img_trans)        
        self.ds_test = ContinuousDS(DM=self.frames["test"], STS=self.series["test"], SCS=self.labels["test"],
            wdw_len=self.wdw_len, wdw_str=self.wdw_str, sts_str=self.sts_str, margin=margin, DM_trans=img_trans)
        
    def train_dataloader(self) -> DataLoader:
        """ Returns the training DataLoader. """
        return DataLoader(self.ds_train, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
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
    