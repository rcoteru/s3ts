from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv

from s3ts.datasets.processing import shift_labels

from collections import Counter
import multiprocessing as mp
from pathlib import Path
import numpy as np
import logging

log = logging.Logger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class ESM(Dataset):

    def __init__(self, 
            OESM: np.ndarray, 
            labels: np.ndarray, 
            window_size: int = 5, 
            windows_label: str = "mode", # "last", "mode"
            transform = None, 
            target_transform = None):

        self.labels = labels
        self.OESM = OESM

        self.window_size = window_size
        self.window_label = windows_label

        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def load_data(path, mode: str):
        
        with np.load(path, 'r') as data:
            if mode == "train":
                labels = data['labels_train']
                OESM = data['OESM_train']
                STS = data['STS_train']
            elif mode == "test":
                labels = data['labels_train']
                OESM = data['OESM_train']
                STS = data['STS_train']

        return OESM, labels, STS

    def __len__(self):

        """ Return the number of samples in the dataset. """
        
        STS_length = len(self.labels)
        n_samples_STS = STS_length + 1 - self.window_size

        return n_samples_STS

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (data, label) from the dataset. """

        STS_length = self.labels.shape[0]
        n_samples_STS = STS_length + 1 - self.window_size

        # sts_idx = idx // n_samples_per_STS
        wdw_idx = idx % n_samples_STS

        labels = self.labels[wdw_idx:wdw_idx + self.window_size]
        window = self.OESM[:, :, wdw_idx:wdw_idx + self.window_size]
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

class ESM_DM(LightningDataModule):

    def __init__(self, 
            target_file: Path, 
            window_size: int, 
            batch_size: int, 
            label_shft: int = 0
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

        log.info("Normalizing data...")
        avg, std = np.average(OESM_train), np.std(OESM_train)
        transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((avg,), (std,))])

        log.info("Creating train dataset...")
        self.ds_train = ESM(OESM=OESM_train, labels=labels_train, window_size=self.window_size,transform=transform)

        log.info("Creating val   dataset...")
        self.ds_val = ESM(OESM=OESM_val, labels=labels_val, window_size=self.window_size, transform=transform)

        log.info("Creating test  dataset...")
        self.ds_test = ESM(OESM=OESM_test, labels=labels_test, window_size=self.window_size, transform=transform)

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
