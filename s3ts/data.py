"""
Data acquisition and preprocessing for the neural network.

@version 2022-12
@author Raúl Coterillo
"""

from __future__ import annotations

# package imports
from s3ts.data_aux import AugProbabilities
from s3ts.data_aux import download_dataset, build_STS, compute_medoids
from s3ts.data_esm import compute_OESM_parallel

# external imports
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv

# numpy / scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import numpy as np

# standard library
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import logging

log = logging.Logger(__name__)

data_folder: str    = "data"
image_folder: str   = "images"

dataset_fname: str = "dataset.npz"
indexes_fname: str = "indexes.npz"

# ========================================================= #
#                    PYTORCH DATASETS                       #
# ========================================================= #

# TODO
class MTaskDataset(Dataset):

    def __init__(self,
            tasks: AuxTasksParams,
            frames: np.ndarray,
            series: np.ndarray,
            olabels: np.ndarray,
            dlabels: np.ndarray,
            indexes: np.ndarray,
            window_size: int,
            transform = None, 
            target_transform = None
            ) -> None:

        self.tasks = tasks
        
        self.frames = frames
        self.series = series
        self.olabels = olabels
        self.dlabels = dlabels
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

        olabel = self.olabels[idx]
        dlabel = self.dlabels[idx]
        dlabel_pred = self.dlabels[idx + self.window_size]
        window = self.frames[:,:, idx-self.window_size:idx]
        
        # TODO not sure if needed anymore
        # adjust for torch-vision indexing
        window = np.moveaxis(window, 0, -1)

        if self.transform:
            window = self.transform(window)
        if self.target_transform:
            label = self.target_transform(label)

        if self.tasks.main_task_only:
            return window, olabel
        else:
            return window, (olabel, dlabel, dlabel_pred)

# ========================================================= #
#                    PYTORCH DATAMODULE                     #
# ========================================================= #

class MTaskDataModule(LightningDataModule):

    experiment: str     # name of the experiment folder
    dataset: str        # name of the UCR/UEA dataset

    sts_length: int     # 
    window_size: int    # size of the 
    batch_size: int     # 

    num_workers: int    # number of workers for the DataLoaders
    random_state: int   # 

    tasks: AuxTasksParams           # 
    aug_probs: AugProbabilities     # 

    def __init__(self, 
            experiment: str,
            dataset: str, 
            sts_length: int,
            window_size: int, 
            tasks: AuxTasksParams,
            # aux_patts: list[function], # TODO add
            batch_size: int,
            random_state: int = 0,
            test_size: float = 0.3,
            rho_memory: float = 0.1,
            extra_samples: int = 2,
            distance_type: str = 'euclidean',
            shuffle_samples: bool = True,
            aug_probs: AugProbabilities = None,
            num_workers: int = mp.cpu_count()//2,
            # patt_length: int = None, # TODO check how pattern downsampling affects classification
            ) -> None:
        
        super().__init__()

        # ~~~~~~~~~~~~~~ input checks ~~~~~~~~~~~~~~ 
        
        self.experiment = experiment
        self.dataset = dataset
        self.tasks = tasks

        if sts_length <= 5:
            raise ValueError(f"Invalid sts_length '{sts_length}'")
        self.sts_length = sts_length 

        if window_size <= 0:
            raise ValueError(f"Invalid window_size '{window_size}'")
        self.window_size = int(window_size)

        if distance_type not in ("euclidean", "dtw"):
            raise NotImplementedError(f"Invalid distance_type '{distance_type}'")
        self.distance_type = distance_type

        if not (0.1 <= test_size <= 0.9):
            raise ValueError(f"Invalid test_size '{test_size}'")
        self.test_size = test_size

        if batch_size not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError(f"Invalid batch_size '{batch_size}'")
        self.batch_size = batch_size
        
        # safety checks
        self.rho_memory = rho_memory

        self.aug_probs = aug_probs
        
        self.num_workers = num_workers
        self.extra_samples = extra_samples
        self.shuffle_samples = shuffle_samples
        self.random_state = random_state 

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

        # create output folder
        self.exp_path = Path.cwd() / data_folder / experiment
        self.exp_path.mkdir(parents=True, exist_ok=True)

        # compute the dataset
        self.__compute_dataset()
        self.n_patterns = self.frames_train.shape[0]

        # generate valid indexes
        self.__generate_indexes()

        # normalization_transform
        avg, std = np.average(self.X_train), np.std(self.X_train)
        transform = tv.transforms.Compose([tv.transforms.ToTensor(), 
                          tv.transforms.Normalize((avg,), (std,))])

        # training dataset
        self.ds_train = MTaskDataset(tasks=tasks,
            frames=self.frames_train, series=self.series_train,
            olabels=self.olabels_train, dlabels=self.dlabels_train,
            indexes=self.indexes_train, window_size=self.window_size,
            transform=transform)

        # validation dataset
        self.ds_eval = MTaskDataset(tasks=tasks,
            frames=self.frames_train, series=self.series_train,
            olabels=self.olabels_train, dlabels=self.dlabels_train,
            indexes=self.indexes_eval, window_size=self.window_size,
            transform=transform)

        # test dataset
        self.ds_test = MTaskDataset(tasks=tasks,
            frames=self.frames_train, series=self.series_train,
            olabels=self.olabels_train, dlabels=self.dlabels_train,
            indexes=self.indexes_test, window_size=self.window_size,
            transform=transform)

        log.info("Patterns:", self.n_patterns)
        log.info("Labels:", self.n_labels)
        log.info("Train samples:", len(self.ds_train))
        log.info("Eval samples:",  len(self.ds_eval))
        log.info("Test samples:",  len(self.ds_test))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __compute_dataset(self, force: bool = False):

        """ Download a dataset from the UCR / UEA archive. """

        save_path = self.exp_path / dataset_fname

        #if computation has already been done, just load it
        if (not force) and save_path.is_file():
            with np.load(save_path, allow_pickle=True) as data:
                self.X, self.Y, self.mapping = data["X"], data["Y"], data["mapping"]
                self.n_labels = int(len(np.unique(self.Y)))
                self.sample_length = self.X.shape[1]
                self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, 
                    test_size=self.test_size, random_state=int(data["random_state"]), stratify=self.Y, shuffle=True)
                self.medoids, self.medoid_ids = data["medoids"], data["medoid_ids"]
                self.series_train, self.olabels_train = data["series_train"], data["olabels_train"]
                self.train_length = len(self.olabels_train)
                self.series_test, self.olabels_test = data["series_test"], data["olabels_test"]
                self.test_length = len(self.olabels_test)
                self.frames_train, self.frames_test = data["frames_train"], data["frames_test"]
            self.__compute_discrete_labels(n_bins=self.tasks.discrete_intervals)
            return
        
        # grab the data from the internet
        self.X, self.Y, self.mapping = download_dataset(self.dataset)
        self.n_labels = int(len(np.unique(self.Y)))
        self.sample_length = self.X.shape[1]

        # train-test split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, 
            test_size=self.test_size, random_state=self.random_state, stratify=self.Y, shuffle=True)

        # compute medoids 
        self.medoids, self.medoid_ids = compute_medoids(self.X_train, self.Y_train, distance_type=self.distance_type)

        # create train STS from samples
        self.series_train, self.olabels_train = build_STS(X=self.X_train, Y=self.Y_train, 
            sts_length=self.extra_samples+self.sts_length, random_state=self.random_state,
            aug_probs=self.aug_probs) #, skip_ids=medoid_ids)
        self.train_length = len(self.olabels_train)

        # create test STS from samples TODO hacer que no sea aleatorio, longitud tamaño test
        self.series_test, self.olabels_test = build_STS(X=self.X_test, Y=self.Y_test, 
            sts_length=self.extra_samples+self.sts_length, random_state=self.random_state,
            aug_probs=None)
        self.test_length = len(self.olabels_test)

        # compute the OESM
        self.frames_train = compute_OESM_parallel(self.series_train, patterns=self.medoids, rho=self.rho_memory)
        self.frames_test = compute_OESM_parallel(self.series_test, patterns=self.medoids, rho=self.rho_memory)

        # save everything to a file
        np.savez_compressed(save_path,
            X=self.X, Y=self.Y, mapping=self.mapping, 
            medoids=self.medoids, medoid_ids=self.medoid_ids,
            series_train=self.series_train, olabels_train=self.olabels_train,
            series_test=self.series_test,   olabels_test=self.olabels_test, 
            frames_train=self.frames_train, frames_test=self.frames_test,
            random_state=self.random_state)

        # calculate discrete labels
        self.__compute_discrete_labels(n_bins=self.tasks.discrete_intervals)
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
    def __compute_discrete_labels(self, n_bins: int) -> None:
        """ Computes discrete labels for the time series. """
        self.kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                    strategy="quantile", random_state=self.random_state)
        self.kbd.fit(self.series_train.reshape(-1,1))
        self.dlabels_train = self.kbd.transform(self.series_train.reshape(-1,1))
        self.dlabels_test = self.kbd.transform(self.series_test.reshape(-1,1))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
    def __generate_indexes(self) -> None:

        """ Generates arrays with valid indexes. """

        start_buff = self.sample_length*self.extra_samples
        
        train_length = self.train_length - start_buff
        test_length = self.test_length - start_buff

        self.indexes_test = np.arange(start_buff, start_buff + test_length//2)
        self.indexes_eval = np.arange(start_buff + test_length//2, self.test_length)
        self.indexes_train = np.arange(start_buff, self.train_length)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def train_dataloader(self):
        """ Returns the training DataLoader. """
        return DataLoader(self.ds_train, batch_size=self.batch_size, 
            shuffle=self.shuffle_samples, num_workers=self.num_workers)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def val_dataloader(self):
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_eval, batch_size=self.batch_size, 
            shuffle=self.shuffle_samples, num_workers=self.num_workers)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def test_dataloader(self):
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            shuffle=self.shuffle_samples, num_workers=self.num_workers)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def predict_dataloader(self):
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            shuffle=self.shuffle_samples, num_workers=self.num_workers)