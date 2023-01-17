# external imports
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv
import torch

import multiprocessing as mp

import numpy as np

# ================================================================= #

class BaseDataset(Dataset):

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
        label = self.labels[idx,:]
        frame = self.frames[:,:,idx - self.window_size:idx]
        
        if self.transform:
            frame = self.transform(frame)
        if self.target_transform:
            label = self.target_transform(label)

        return frame, label

# ================================================================= #

class BaseDataModule(LightningDataModule):

    def __init__(self,
            # calculate this outside
            STS: np.ndarray,
            DFS: np.ndarray,
            labels: np.ndarray,
            # ~~~~~~~~~~~~~~~~~~~~~~
            window_size: int, 
            batch_size: int,
            random_state: int = 0,
            eval_size: float = 0.15,
            test_size: float = 0.15
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
        self.random_state = random_state 
        self.num_workers = mp.cpu_count()//2

        # generate train/eval/test indexes
        indexes = np.arange(self.window_size*3, self.l_DFS)
        br1, br2 = int(len(indexes)*(1-test_size-eval_size)), int(len(indexes)*(1-test_size))
        self.train_idx, self.eval_idx, self.test_idx = indexes[:br1], indexes[br1:br2], indexes[br2:]

        # normalization_transform
        transform = tv.transforms.Normalize(
                self.DFS[:,:,self.window_size*3:br1].mean(axis=[1,2]), 
                self.DFS[:,:,self.window_size*3:br1].std(axis=[1,2]))

        # training dataset
        self.ds_train = BaseDataset(indexes=self.train_idx,
            frames=self.DFS, series=self.STS, labels=self.labels, window_size=self.window_size, transform=transform)

        self.ds_eval = BaseDataset(indexes=self.eval_idx,
            frames=self.DFS, series=self.STS, labels=self.labels, window_size=self.window_size, transform=transform)

        self.ds_test = BaseDataset(indexes=self.test_idx,
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