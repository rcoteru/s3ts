"""
Data acquisition and preprocessing for the neural network.

@version 2022-12
@author Raúl Coterillo
"""

# torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv

# numpy
import numpy as np

# standard library
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import multiprocessing as mp
from pathlib import Path
import logging

from s3ts import RANDOM_STATE
rng = np.random.default_rng(seed=RANDOM_STATE)
log = logging.Logger(__name__)

@dataclass
class AuxTasksParams:

    """ Settings for the auxiliary tasks. """

    main_task_only: bool = False    # ¿Only do the main task?
    
    disc: bool = True               # Discretized clasification
    disc_intervals: bool = 10       # Discretized intervals

    pred: bool = True               # Prediction
    pred_time: int = None           # Prediction time (if None, then window_size)
    
    aenc: bool = True               # Autoencoder

# ========================================================= #
#                    PYTORCH DATASETS                       #
# ========================================================= #

class ScalarLabelsDataset(Dataset):

    def __init__(self, 
            file: Path,
            window_size: int,
            label_choice: str,     # "last" / "mode"
            transform = None, 
            target_transform = None
            ) -> None:

        """ Reads files. """

        self.file = file
        self.window_size = window_size
        self.label_choice = label_choice
        self.transform = transform
        self.target_transform = target_transform

        with np.load(file, 'r') as data: 
            self.X_frames = data["frames"]
            self.X_series = data["series"]
            self.Y_values = data["values"]

        self.STS_length = len(self.Y_values)
        self.n_samples = self.STS_length + 1 - self.window_size

    def __len__(self) -> int:

        """ Return the number of samples in the dataset. """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (x, y) from the dataset. """

        wdw_idx = idx % self.n_samples

        labels = self.labels[wdw_idx:wdw_idx + self.window_size]
        window = self.OESM[:, :, wdw_idx:wdw_idx + self.window_size]
        
        # adjust for torch-vision indexing
        window = np.moveaxis(window, 0, -1)

        if self.window_label == "last":     # pick the last label of the window
            label = labels.squeeze()[-1]   
        elif self.window_label == "mode":   # pick the most common label in the window
            label_counts = dict(Counter(labels))
            label = int(max(label_counts, key=label_counts.get))
        else:
            raise NotImplementedError

        if self.transform:
            window = self.transform(window)
        if self.target_transform:
            label = self.target_transform(label)

        return x, y

# ========================================================= #
#                  PYTORCH DATAMODULE                       #
# ========================================================= #

class STSDataModule(LightningDataModule):

    def __init__(self, 
            dataset_name: str, 
            window_size: int, 
            aux_patterns: list[function],
            tasks: AuxTasksParams,
            batch_size: int, 
            patt_length: int = None,

            ) -> None:
        
        super().__init__()


        self.target_file = target_file
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_workers = mp.cpu_count()//2

        log.info(" ~ PREPARING DATA MODULE ~ ")

        log.info("Loading data...")
        OESM_train, labels_train,  STS_train = ESM.load_data(self.target_file, mode="train")
        OESM_test, labels_test,  STS_test = ESM.load_data(self.target_file, mode="test")

        # shift labels if needed
        labels_train, OESM_train, STS_train = shift_labels(labels_train, OESM_train, STS_train, shift=label_shft)
        labels_test, OESM_test, STS_test = shift_labels(labels_test, OESM_test, STS_test, shift=label_shft)

        log.info("Splitting validation set from test data...")
        STS_length = len(labels_test)
        OESM_val , labels_val =  OESM_test[:,:,STS_length//2:], labels_test[STS_length//2:]
        OESM_test, labels_test = OESM_test[:,:,:STS_length//2], labels_test[:STS_length//2]

        self.labels_size = len(np.unique(labels_train)) # number of labels
        self.channels = OESM_train.shape[0]             # number of patterns

        log.info("       Train shape:", OESM_train.shape)
        log.info("         Val shape:", OESM_val.shape)
        log.info("        Test shape:", OESM_test.shape)
        log.info("  Number of labels:", self.labels_size)
        log.info("Number of patterns:", self.channels)

        # log.info("Normalizing data...")
        # avg, std = np.average(OESM_train), np.std(OESM_train)
        # transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((avg,), (std,))])

        # log.info("Creating train dataset...")
        # self.ds_train = ESM(OESM=OESM_train, labels=labels_train, window_size=self.window_size,transform=transform)

        # log.info("Creating val   dataset...")
        # self.ds_val = ESM(OESM=OESM_val, labels=labels_val, window_size=self.window_size, transform=transform)

        # log.info("Creating test  dataset...")
        # self.ds_test = ESM(OESM=OESM_test, labels=labels_test, window_size=self.window_size, transform=transform)

        # log.info(" ~ DATA MODULE PREPARED ~ ")  

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)