"""
Automation of the data manipulation tasks.
"""

# import library functions / objects
from s3ts.datasets.processing import acquire_dataset, compute_medoids, build_STS, discretize_TS
from s3ts.datasets.oesm import compute_OESM_parallel
from s3ts.datasets.modules import ESM_DM

# import constants
from s3ts.tasks import data_folder, base_main_fname, base_task_fname
from s3ts import RANDOM_STATE

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import logging

log = logging.Logger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def prepare_experiment(exp_name: str, exists_ok: bool = True) -> Path:

    save_path = Path.cwd() / data_folder / exp_name
    log.info(f"Creating experiment '{exp_name}' in {save_path}")
    save_path.mkdir(parents=True, exist_ok=exists_ok)

    return save_path

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def prepare_dataset(
        exp_path: Path, 
        dataset: str, 
        test_size: float,
        medoid_type: str = "dtw", # "dtw" or "euclidean",
        force: bool = False
        ) -> None:

    log.info(f"Preparing dataset '{dataset}' in '{exp_path.name}' ...")
    base_file = exp_path / base_main_fname

    # exit if calculation not needed
    if not (force or not base_file.exists()):
        return None

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def prepare_classification_data(
        exp_path: Path,
        task_name: str,
        sts_length: int,
        label_type: str,
        label_shft: int,
        rho_memory: float = 0.1,
        batch_size: int = 128,
        wndow_size: int = 5,
        force: bool = False
        ) -> ESM_DM:

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

    return ESM_DM(target_file, window_size=wndow_size, batch_size=batch_size, label_shft=label_shft)
