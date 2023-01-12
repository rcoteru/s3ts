# external imports
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv

import torch

import numpy as np

class SimpleDataset(Dataset):

    def __init__(self,
            frames: np.ndarray,
            series: np.ndarray,
            labels: np.ndarray,
            indexes: np.ndarray,
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

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """ Return the number of samples in the dataset. """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (x, y) from the dataset. """

        idx = self.indexes[idx]

        label = self.labels[idx]
        frame = self.frames[:,:,idx - self.window_size:idx]
        
        # TODO not sure if needed anymore
        # adjust for torch-vision indexing
        window = np.moveaxis(window, 0, -1)

        if self.transform:
            window = self.transform(window)
        if self.target_transform:
            label = self.target_transform(label)

        return window, (olabel, dlabel, dlabel_pred)