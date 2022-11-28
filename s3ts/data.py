"""
Data acquisition and preprocessing for the neural network.

@version 2022-12
@author Raúl Coterillo
"""

# standard library
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import multiprocessing as mp
from pathlib import Path
import logging

# external imports
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision as tv
import numpy as np

# package imports
from s3ts.data_aux import acquire_dataset

from s3ts import RANDOM_STATE
rng = np.random.default_rng(seed=RANDOM_STATE)
log = logging.Logger(__name__)

data_folder = "data"
image_folder = "images"

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

        return window, labels

# ========================================================= #
#                  PYTORCH DATAMODULE                       #
# ========================================================= #

class STSDataModule(LightningDataModule):

    def __init__(self, 
            experiment_name: str,
            dataset_name: str, 
            window_size: int, 
            aux_patterns: list[function],
            tasks: AuxTasksParams,
            batch_size: int, 
            patt_length: int = None,
            ) -> None:
        
        super().__init__()


        # create output folder
        save_path = Path.cwd() / data_folder / exp_name
        log.info(f"Creating experiment '{exp_name}' in {save_path}")
        save_path.mkdir(parents=True, exist_ok=exists_ok)

        # assign whatever variables
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

    def __download_dataset(self):

        # grab the data 
        X, Y, mapping = acquire_dataset(dataset)
        n_labels = len(np.unique(Y))

        # train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, 
                random_state=RANDOM_STATE, stratify=Y, shuffle=True)

        # compute medoids 
        medoids, medoid_ids = compute_medoids(X_train, Y_train, distance_type=medoid_type)
        
        # plot the medoids
        img_folder = exp_path / "images" / "medoids"
        img_folder.mkdir(parents=True, exist_ok=True)
        for i in range(medoids.shape[0]):
            plt.figure(figsize=(6,6))
            plt.title(f"Medoid of class {i}")
            plt.plot(medoids[i])
            plt.savefig(img_folder / f"medoid{i}.png")
            plt.close()

        # save everything to a file
        np.savez_compressed(base_file, 
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test,
            medoids=medoids, medoid_ids=medoid_ids,
            mapping=mapping, test_size=test_size, 
            n_labels=n_labels)

    def __generate_shared_data(
            exp_path: Path,
            task_name: str,
            sts_length: int,
            label_type: str,
            label_shft: int,
            rho_memory: float = 0.1,
            batch_size: int = 128,
            wndow_size: int = 5,
            force: bool = False
            ) -> None:

        log.info(f"Generating classification data for task '{task_name}' in '{exp_path.name}' ...")
        target_file = exp_path / f"{task_name}_{base_task_fname}"

        # load datamodule if calculation is not needed
        if not (force or not target_file.exists()):
            return ESM_DM(target_file, window_size=wndow_size, batch_size=batch_size, label_shft=label_shft)

        # load experiment data
        base_file = exp_path / base_main_fname
        with np.load(base_file) as data:
            medoids, medoid_ids = data["medoids"], data["medoid_ids"]
            X_train, Y_train = data["X_train"], data["Y_train"]
            X_test,  Y_test  = data["X_test"], data["Y_test"]
            n_labels = data["n_labels"]

        # calculate labels
        if label_type == "original":
            Y_train = Y_train
            Y_test = Y_test
        elif label_type == "discrete_STS":
            Y_train, kbd = discretize_TS(X_train, intervals=int(n_labels), strategy="quantile")
            
            # kbd = KBinsDiscretizer(n_bins=intervals, encode="ordinal",
            # strategy=strategy, random_state=random_state)
            # kbd.fit(TS)
            
            # return kbd.transform(TS)

            Y_test = kbd.transform(X_test)
        else:
            raise NotImplementedError(f"Unknown label type: '{label_type}'")

        # create train STS from samples
        STS_train, labels_train = build_STS(X=X_train, Y=Y_train, 
            sts_length=sts_length, skip_ids=medoid_ids,
            aug_jitter = 0, aug_time_warp = 0, 
            aug_scaling = 0, aug_window_warp = 0) 
        
        # create test STS from samples
        # TODO hacer que no sea aleatorio, longitud tamaño test
        STS_test, labels_test = build_STS(X=X_train, Y=Y_train, 
            sts_length=sts_length)   

        # compute the OESM
        OESM_train = compute_OESM_parallel(STS_train, patterns = medoids, rho=rho_memory)
        OESM_test = compute_OESM_parallel(STS_test, patterns = medoids, rho=rho_memory)

        # save the data
        np.savez_compressed(target_file, 
            STS_train = STS_train, STS_test = STS_test,
            labels_train = labels_train, labels_test = labels_test,
            OESM_train = OESM_train, OESM_test = OESM_test)

    def __generate_discrete_labels():


        pass

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)