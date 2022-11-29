"""
Data acquisition and preprocessing for the neural network.

@version 2022-12
@author Raúl Coterillo
"""

# package imports
from s3ts.data_aux import acquire_dataset, build_STS, compute_medoids
from s3ts.data_aux import AugProbabilities

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
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import multiprocessing as mp
from pathlib import Path
import logging

log = logging.Logger(__name__)

data_folder: str    = "data"
image_folder: str   = "images"

dataset_fname: str = "dataset.npz"
indexes_fname: str = "indexes.npz"

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

# TODO
class MTaskDataset(Dataset):

    def __init__(self,
            frames: np.ndarray,
            series: np.ndarray,
            olabels: np.ndarray,
            dlabels: np.ndarray,
            indexes: np.ndarray,
            window_size: int,
            label_choice: str,      # "last" / "mode"
            transform = None, 
            target_transform = None
            ) -> None:

        """ Reads files. """

        if label_choice not in ("last", "mode"):
            raise NotImplementedError(f"Invalid label_choice '{label_choice}'.")
        self.label_choice = label_choice
        
        self.frames = frames
        self.series = series
        self.olabels = olabels
        self.dlabels = dlabels
        self.indexes = indexes

        self.n_samples = len(self.indexes)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """ Return the number of samples in the dataset. """
        return self.n_samples

    # TODO
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

        return window, labels

# ========================================================= #
#                  PYTORCH DATAMODULE                       #
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
            n_bins: int,
            window_size: int, 
            tasks: AuxTasksParams,
            # aux_patts: list[function], # TODO add
            batch_size: int,
            random_state: int = 0,
            test_size: int = 0.3,
            rho_memory: float = 0.1,
            
            label_choice: str = 'last',
            distance_type: str = 'euclidean',
            aug_probs: AugProbabilities = None,
            num_workers: int = mp.cpu_count()//2,
            # patt_length: int = None, # TODO check how pattern downsampling affects classification
            ) -> None:
        
        super().__init__()

        # ~~~~~~~~~~~~~~ input checks ~~~~~~~~~~~~~~ 
        
        self.experiment = experiment   
        self.dataset = dataset

        if sts_length <= 5:
            raise ValueError(f"Invalid window size '{window_size}'")
        self.sts_length = sts_length 

        if sts_length <= 5:
            raise ValueError(f"Invalid window size '{window_size}'")
        self.nbins = n_bins

        if window_size <= 0:
            raise ValueError(f"Invalid window size '{window_size}'")
        self.window_size = int(window_size)

        self.distance_type = distance_type

        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state 

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

        # create output folder
        self.exp_path = Path.cwd() / data_folder / experiment
        # log.info(f"Creating experiment '{exp_name}' in {save_path}")
        self.exp_path.mkdir(parents=True, exist_ok=False)

        # compute the dataset
        self.__compute_dataset()

        # generate 
        self.__generate_indexes()

        # normalization_transform
        # log.info("Normalizing data...")
        avg, std = np.average(self.X_train), np.std(self.X_train)
        transform = tv.transforms.Compose([tv.transforms.ToTensor(), 
                          tv.transforms.Normalize((avg,), (std,))])

        # log.info("Splitting validation set from test data...")2
        test_indexes = np.arange(self.test_length//2)
        eval_indexes = np.arange(self.test_length//2, self.test_length)

        self.ds_train = MTaskDataset(
            frames=self.frames_train, series=self.series_train,
            olabels=self.olabels_train, dlabels=self.dlabels_train,
            indexes=self.indexes_train, window_size=self.window_size,
            label_choice=self.label_choice, transform=transform)

        self.ds_eval = MTaskDataset(
            frames=self.frames_train, series=self.series_train,
            olabels=self.olabels_train, dlabels=self.dlabels_train,
            indexes=self.indexes_train, window_size=self.window_size,
            label_choice=self.label_choice, transform=transform)

        self.ds_test = MTaskDataset(
            frames=self.frames_train, series=self.series_train,
            olabels=self.olabels_train, dlabels=self.dlabels_train,
            indexes=self.indexes_train, window_size=self.window_size,
            label_choice=self.label_choice, transform=transform)

        log.info("Patterns:", self.channels)
        log.info("Labels:", self.labels_size)
        log.info("Train samples:", len(self.ds_train))
        log.info("Eval samples:",  len(self.ds_eval))
        log.info("Test samples:",  len(self.ds_test))

    def __compute_dataset(self):

        """ Download a dataset from the UCR / UEA archive. """

        ## checks if new computation is needed

        # grab the data from the internet
        X, Y, mapping = acquire_dataset(self.dataset)
        self.n_labels = int(len(np.unique(Y)))

        # train-test split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, 
            test_size=self.test_size, random_state=self.random_state, stratify=Y, shuffle=True)

        # compute medoids 
        medoids, medoid_ids = compute_medoids(self.X_train, self.Y_train, distance_type=self.distance_type)
        
        # plot the medoids
        # img_folder = exp_path / "images" / "medoids"
        # img_folder.mkdir(parents=True, exist_ok=True)
        # for i in range(medoids.shape[0]):
        #     plt.figure(figsize=(6,6))
        #     plt.title(f"Medoid of class {i}")
        #     plt.plot(medoids[i])
        #     plt.savefig(img_folder / f"medoid{i}.png")
        #     plt.close()

        # create train STS from samples
        self.series_train, self.olabels_train = build_STS(X=self.X_train, Y=self.Y_train, 
            sts_length=self.sts_length, aug_probs=self.aug_probs) #, skip_ids=medoid_ids)
        self.train_length = len(self.olabels_train)

        # create test STS from samples TODO hacer que no sea aleatorio, longitud tamaño test
        self.series_test, self.olabels_test = build_STS(X=self.X_test, Y=self.Y_test, 
            sts_length=self.sts_length, aug_probs=None)
        self.test_length = len(self.olabels_test)

        # calculate discrete labels
        self.__calculate_discrete_labels(n_bins=self.n_bins)

        # compute the OESM
        self.frames_train = compute_OESM_parallel(STS_train, patterns=medoids, rho=rho_memory)
        self.frames_test = compute_OESM_parallel(STS_test, patterns=medoids, rho=rho_memory)

        # save everything to a file
        np.savez_compressed(self.dataset_fname:
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test,
            medoids=medoids, medoid_ids=medoid_ids,
            mapping=mapping, test_size=test_size, 
            n_labels=n_labels)

        # save the data
        np.savez_compressed(target_file, 
            STS_train = STS_train, STS_test = STS_test,
            labels_train = labels_train, labels_test = labels_test,
            OESM_train = OESM_train, OESM_test = OESM_test)
        
    def __calculate_discrete_labels(self, n_bins: int) -> None:
        self.kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal",
                    strategy="quantile", random_state=self.random_state)
        self.kbd.fit(self.series_train)
        self.dlabels_train = self.kbd.transform(self.series_train)
        self.dlabels_test = self.kbd.transform(self.series_test)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)