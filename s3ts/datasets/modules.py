from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision as tv

from sklearn.model_selection import train_test_split

from collections import Counter
import multiprocessing as mp
from pathlib import Path
import numpy as np
import logging

log = logging.Logger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class ESM(Dataset):

    def __init__(self, 
            ESMs: np.ndarray, 
            labels: np.ndarray, 
            window_size: int = 5, 
            windows_label: str = "mode", # "last", "mode"
            transform = None, 
            target_transform = None):

        self.labels = labels
        self.ESMs = ESMs

        self.window_size = window_size
        self.window_label = windows_label

        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def load_data(path):
        with np.load(path, 'r') as data:
            labels = data['labels']
            ESMs = data['ESMs']
            STSs = data['STSs']
        return ESMs, labels, STSs

    def __len__(self):

        """ Return the number of samples in the dataset. """
        
        number_of_STSs = self.labels.shape[0]
        STS_length = self.labels.shape[1]
        n_samples_per_STS = STS_length + 1 - self.window_size

        return n_samples_per_STS * number_of_STSs

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (data, label) from the dataset. """

        STS_length = self.labels.shape[1]
        n_samples_per_STS = STS_length + 1 - self.window_size

        sts_idx = idx // n_samples_per_STS
        wdw_idx = idx % n_samples_per_STS

        labels = self.labels[sts_idx, wdw_idx:wdw_idx + self.window_size]
        window = self.ESMs[sts_idx, :, :, wdw_idx:wdw_idx + self.window_size]
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

        return window, label

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class ESM_DM(pl.LightningDataModule):

    def __init__(self, 
            target_file: Path, 
            window_size: int, 
            batch_size: int, 
            ) -> None:
        
        super().__init__()
        self.target_file = target_file
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_workers = mp.cpu_count()//2

        log.info(" ~ PREPARING DATA MODULE ~ ")

        log.info("Loading data...")
        ESMs_train, labels_train, _ = ESM.load_data(self.data_path / f"DB-train_{self.task}.npz")
        ESMs_test, labels_test, _ = ESM.load_data(self.data_path / f"DB-test_{self.task}.npz")

        log.info("Splitting validation set from test data...")

        n_samples_test = ESMs_test.shape[0]
        val_idxs, test_idxs = train_test_split(range(n_samples_test), train_size=0.5, test_size=0.5)

        ESMs_val , labels_val = ESMs_test[val_idxs], labels_test[val_idxs]
        ESMs_test, labels_test = ESMs_test[test_idxs], labels_test[test_idxs]

        self.labels_size = len(np.unique(labels_train)) # number of labels
        self.channels = ESMs_train.shape[1]             # number of patterns

        log.info("       Train shape:", ESMs_train.shape)
        log.info("         Val shape:", ESMs_val.shape)
        log.info("        Test shape:", ESMs_test.shape)
        log.info("  Number of labels:", self.labels_size)
        log.info("Number of patterns:", self.channels)

        log.info("Normalizing data...")
        avg, std = np.average(ESMs_train), np.std(ESMs_train)
        transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((avg,), (std,))])

        log.info("Creating train dataset...")
        self.ds_train = ESM(ESMs=ESMs_train, labels=labels_train, window_size=self.window_size,transform=transform)

        log.info("Creating val   dataset...")
        self.ds_val = ESM(ESMs=ESMs_val, labels=labels_val, window_size=self.window_size, transform=transform)

        log.info("Creating test  dataset...")
        self.ds_test = ESM(ESMs=ESMs_test, labels=labels_test, window_size=self.window_size, transform=transform)

        log.info(" ~ DATA MODULE PREPARED ~ ")        

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
