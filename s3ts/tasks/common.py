
from s3ts.datasets.processing import acquire_dataset, compute_medoids, build_STSs
from s3ts import RANDOM_STATE

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

data_folder = "data"
base_name = "base_data.npz"

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

    base_file = exp_path / base_name

    # exit if calculation not needed
    if not (force or not base_file.exists()):
        return None

    # grab the data 
    X, Y, mapping = acquire_dataset(dataset)

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
    np.savez_compressed(save_file, 
        X_train=X_train, Y_train=Y_train,
        X_test=X_test, Y_test=Y_test,
        medoids=medoids, medoid_ids=medoid_ids,
        mapping=mapping, test_size=test_size)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def prepare_classification_data(
            exp_path: Path,
            task_name: str,
            sts_length: int,
            force: bool = False
            ) -> None:

    log.info(f"Generating classification data for task '{task_name}' in '{exp_path.name}' ...")

    target_file = exp_path / f"{task_name}_data.npz"

    # exit if calculation not needed
    if not (force or not target_file.exists()):
        return None

    base_file = exp_path / base_name

    with np.load(base) as data:
        X, Y, mapping = data["X"], data["Y"], data["mapping"]

    save_file_train = save_path / "DB-train_main-task.npz"
    save_file_test = save_path / "DB-test_main-task.npz"

    n_test_sts = round(STS_NUMBER*TEST_SIZE)
    n_train_sts = STS_NUMBER - n_test_sts

    # create STSs from samples
    STSs_X_train, STSs_Y_train = build_STSs(X=X_train, Y=Y_train, samples_per_sts=STS_LENGTH, 
        number_of_sts=n_train_sts, skip_ids=medoid_ids) 
    STSs_X_test, STSs_Y_test = build_STSs(X=X_train, Y=Y_train, samples_per_sts=STS_LENGTH, 
        number_of_sts=n_test_sts, skip_ids=medoid_ids)   

    # compute the ESMs
    ESMs_train = compute_OESMs(STSs_X_train, medoids, rho=RHO, nprocs=NPROCS)
    ESMs_test = compute_OESMs(STSs_X_test, medoids, rho=RHO, nprocs=NPROCS)

    # save the data
    np.savez_compressed(save_file_train, STSs=STSs_X_train, labels=STSs_Y_train, ESMs=ESMs_train)
    np.savez_compressed(save_file_test, STSs=STSs_X_test, labels=STSs_Y_test, ESMs=ESMs_test)