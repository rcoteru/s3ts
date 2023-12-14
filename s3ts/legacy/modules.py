#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Data modules for the S3TS project. """

# torch / lightning imports
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch

# standard library imports
import multiprocessing as mp
import logging as log
import numpy as np


class DFDataset(Dataset):

    """ Dataset for the experiments. """

    def __init__(self,
            DM: torch.Tensor,
            STS: torch.Tensor,
            SCS: torch.Tensor,
            index: np.ndarray,
            window_length: int,
            window_time_stride: int,
            window_patt_stride: int,
            DM_transform = None,
            STS_transform = None, 
            SCS_transform = None,
            stride_series: bool = True,
            ) -> None:

        """ Initialize the dataset. 
        
        Parameters
        ----------
        DM : torch.Tensor
            Dissimilarity matrix
        STS : torch.Tensor
            Streaming Time Series
        SCS : torch.Tensor
            Streaming Class Series
        index : np.ndarray
            Index of the samples
        window_length : int
            Length of the window
        window_time_stride : int
            Time stride of the frame window
        window_patt_stride : int
            Pattern stride of the frame window
        DM_transform : callable, optional
            Transformation to apply to the DM
        STS_transform : callable, optional
            Transformation to apply to the STS
        SCS_transform : callable, optional
            Transformation to apply to the SCS
        """

        super().__init__()

        self.DM = DM
        self.STS = STS
        self.SCS = SCS
        self.index = index

        self.window_length = window_length
        self.wts = window_time_stride
        self.wps = window_patt_stride
        self.stride_series = stride_series

        self.available_events = 1

        self.DM_transform = DM_transform
        self.STS_transform = STS_transform
        self.SCS_transform = SCS_transform

    def __len__(self) -> int:

        """ Return the length of the dataset. """
        
        return len(self.index)


    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:

        """ Return an entry (x, y) from the dataset. """

        idx = self.index[idx]
       
        # Grab the frame
        frame = self.DM[:,::self.wps,idx - self.window_length*self.wts+1:idx+1:self.wts]
        if self.DM_transform:
            frame = self.DM_transform(frame)

        # Grab the series
        if self.stride_series:
            series = self.STS[idx - self.window_length*self.wts+1:idx+1:self.wts]
        else:
            series = self.STS[idx - self.window_length*self.wts+1:idx+1]
        if self.STS_transform:
            series = self.STS_transform(series)

        # Grab the label
        label = self.SCS[idx]
        if self.SCS_transform:
            label = self.SCS_transform(label)

        # Return the frame, series, and label
        return frame, series, label

# ================================================================= #

class DFDataModule(LightningDataModule):

    """ Data module for the experiments. """

    def __init__(self,
            X_train: np.ndarray, X_test: np.ndarray,
            Y_train: np.ndarray, Y_test: np.ndarray,
            STS_train: np.ndarray, STS_test: np.ndarray, 
            SCS_train: np.ndarray, SCS_test: np.ndarray,
            DM_train: np.ndarray, DM_test: np.ndarray,
            event_length: int, patterns: np.ndarray, 
            val_size: float, batch_size: int,
            window_length: int, 
            window_time_stride: int, 
            window_patt_stride: int, 
            stride_series: bool, 
            random_state: int = 0, 
            num_workers: int = mp.cpu_count()//2
            ) -> None:
        
        """ Initialize the DFs data module.
        
        Parameters
        ----------
        STS_tra : np.ndarray
            Training Streaming Time Series
        STS_pre : np.ndarray
            Pre-training Streaming Time Series
        SCS_tra : np.ndarray
            Training Streaming Class Series
        SCS_pre : np.ndarray
            Pre-training Streaming Class Series
        DM_tra : np.ndarray
            Training Dissimilarity Matrix
        DM_pre : np.ndarray
            Pre-training Dissimilarity Matrix
        STS_train_events : int
            Number of events in the training STS
        STS_pret_events : int
            Number of events in the pretraining STS
        STS_test_events : int
            Number of events in the test STS
        event_length : int
            Length of the events
        patterns : np.ndarray
            Patterns for the DMs
        batch_size : int
            Batch size for the dataloaders
        window_length : int
            Length of the window for the frame
        window_time_stride : int
            Time stride of the frame window
        window_patt_stride : int
            Pattern stride of the frame window
        stride_series : bool
            Whether to stride the series
        random_state : int, optional
            Random state for the data module
        num_workers : int, optional
            Number of workers for the dataloaders
        """
        
        super().__init__()

        # Register dataset parameters
        self.val_size = val_size
        self.batch_size = batch_size
        self.window_length = window_length
        self.window_time_stride = window_time_stride
        self.window_patt_stride = window_patt_stride
        self.stride_series = stride_series
        self.random_state = random_state
        self.num_workers = num_workers

        # Is there a test set?
        self.test = (X_test is not None and Y_test is not None and \
            STS_test is not None and SCS_test is not None and DM_test is not None)

        # Gather dataset info
        self.l_events = event_length    
        self.n_classes = len(np.unique(SCS_train))
        self.n_patterns = patterns.shape[0]
        self.l_patterns = patterns.shape[1]

        # Register datasets
        self.X_train = torch.from_numpy(X_train).to(torch.float32)
        self.Y_train = torch.from_numpy(Y_train).to(torch.int64)
        self.patterns = torch.from_numpy(patterns).to(torch.float32)

        self.STS_train_events = len(STS_train) // self.l_events
        self.STS_train = torch.from_numpy(STS_train).to(torch.float32)
        self.SCS_train = torch.from_numpy(SCS_train).to(torch.int64)
        self.labels_tra = torch.nn.functional.one_hot(self.SCS_train, num_classes=self.n_classes)
        self.DM_train = torch.from_numpy(DM_train).to(torch.float32)

        if self.test:
            self.X_test = torch.from_numpy(X_test).to(torch.float32)
            self.Y_test = torch.from_numpy(Y_test).to(torch.int64)
            self.STS_test_events = len(STS_test) // self.l_events
            self.STS_test = torch.from_numpy(STS_test).to(torch.float32)
            self.SCS_test = torch.from_numpy(SCS_test).to(torch.int64)
            self.labels_test = torch.nn.functional.one_hot(self.SCS_test, num_classes=self.n_classes)
            self.DM_test = torch.from_numpy(DM_test).to(torch.float32)

        # Create the indices
        self.create_sample_index()
        self.create_datasets()

        # Do some logging
        log.info(f"Events in training STS: {self.STS_train_events}")
        log.info(f"Frames in training STS: {len(self.train_indices)} ")
        self.tra_ratios = np.unique(SCS_train[self.train_indices], return_counts=True)[1]/(self.STS_train_events*self.l_events)
        log.info(f"Train STS class ratios: {self.tra_ratios}")

        if self.test:
            log.info(f"Events in testing STS: {self.STS_test_events}")
            log.info(f"Frames in testing STS: {len(self.test_indices)}")
            self.test_ratios = np.unique(SCS_test[self.test_indices], return_counts=True)[1]/(self.STS_test_events*self.l_events)
            log.info(f"Test STS class ratios: {self.test_ratios}")

        # Calculate the memory usage of the datasets
        self.DM_mem = self.DM_train.element_size()*self.DM_train.nelement()
        self.STS_mem = self.STS_train.element_size()*self.STS_train.nelement()
        self.SCS_mem = self.SCS_train.element_size()*self.SCS_train.nelement()
        if self.test:
            self.DM_mem += self.DM_test.element_size()*self.DM_test.nelement()
            self.STS_mem += self.STS_test.element_size()*self.STS_test.nelement()
            self.SCS_mem += self.SCS_test.element_size()*self.SCS_test.nelement()

        log.info(f"DM  memory usage: {self.DM_mem/1e6} MB")
        log.info(f"STS memory usage: {self.STS_mem/1e6} MB")
        log.info(f"SCS memory usage: {self.SCS_mem/1e6} MB")

    def create_sample_index(self, 
            window_length: int = None,
            stride_series: int = None,
            window_time_stride: int = None,
            window_patt_stride: int = None):
        
        """ Create the sample indeces for the datasets. """

        # Check if the window length is set
        if window_length is not None:
            # Check if the window length is in the correct range
            assert window_length > 0, "Window length must be larger than 0"
            self.window_length = window_length

        # Check if the stride series is set
        if stride_series is not None:           
            self.stride_series = stride_series

        # Check if the window time stride is set
        if window_time_stride is not None:
            # Check if the window time stride is in the correct range
            assert window_time_stride > 0, "Window time stride must be larger than 0"
            self.window_time_stride = window_time_stride
        
        # Check if the window pattern stride is set
        if window_patt_stride is not None:
            # Check if the window pattern stride is in the correct range
            assert window_patt_stride > 0, "Window pattern stride must be larger than 0"
            self.window_patt_stride = window_patt_stride

        # Calculate the margin due to the window length
        margin = self.window_length*self.window_time_stride+1

        # Creathe the default indices
        self.train_indices = np.arange(margin, self.STS_train_events*self.l_events)
        if self.test:
            self.test_indices = np.arange(margin, self.STS_test_events*self.l_events)

    def create_datasets(self, val_size: float = None) -> None:

        """ Create/update the datasets for training, validation and testing. """

        # Check if the validation size is set
        if val_size is not None:
            # Check if the validation size is in the correct range
            assert val_size > 0 and val_size < 1, "Validation size must be between 0 and 1"
            self.val_size = val_size

        # Normalization transform for the frames
        DM_transform = tv.transforms.Normalize(
            self.DM_train.mean(axis=[1,2]),
            self.DM_train.std(axis=[1,2]))
        
        # Create the training datasets
        tra_tot_samples = len(self.train_indices)
        tra_train_samples = tra_tot_samples-int(tra_tot_samples*self.val_size)
        self.ds_tra_train = DFDataset(index=self.train_indices[:tra_train_samples],
            DM=self.DM_train, STS=self.STS_train, SCS=self.labels_tra,
            window_length=self.window_length, stride_series=self.stride_series, 
            window_time_stride=self.window_time_stride, window_patt_stride=self.window_patt_stride, 
            DM_transform=DM_transform)
        self.ds_tra_val   = DFDataset(index=self.train_indices[tra_train_samples:],
            DM=self.DM_train, STS=self.STS_train, SCS=self.labels_tra,
            window_length=self.window_length, stride_series=self.stride_series, 
            window_time_stride=self.window_time_stride, window_patt_stride=self.window_patt_stride, 
            DM_transform=DM_transform)

        # Create the testing dataset
        if self.test:
            self.ds_test = DFDataset(index=self.test_indices,
                DM=self.DM_test, STS=self.STS_test, SCS=self.labels_test,
                window_length=self.window_length, stride_series=self.stride_series,
                window_time_stride=self.window_time_stride, window_patt_stride=self.window_patt_stride,
                DM_transform=DM_transform)

    def update_properties(self,
            val_size: float = None, 
            window_length: int = None,
            stride_series: bool = None,
            window_time_stride: int = None,
            window_patt_stride: int = None):
        
        """ Updates the samples index based on the available events. """
        
        self.create_sample_index(window_length=window_length, stride_series=stride_series,
            window_time_stride=window_time_stride, window_patt_stride=window_patt_stride)
        self.create_datasets(val_size=val_size)

    def train_dataloader(self) -> DataLoader:
        """ Returns the training DataLoader. """
        return DataLoader(self.ds_tra_train, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_tra_val, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        if self.test:
            return DataLoader(self.ds_test, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=False,
                pin_memory=True, persistent_workers=True)
        raise ValueError("Test dataset not available")

    def predict_dataloader(self):
        """ Returns the predict DataLoader."""
        return self.test_dataloader()