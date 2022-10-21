"""

@author Raúl Coterillo
"""


# data processing
from s3ts.datasets.processing import acquire_dataset, compute_medoids, build_STSs
from s3ts.datasets.processing import discretize_STSs, shift_STSs_labels
from s3ts.datasets.oesm import compute_OESMs

from sklearn.model_selection import train_test_split

# plotting
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import logging
import sys

log = logging.Logger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s')

# Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# DATASET_NAME = "GunPoint"
DATASET_NAME = "CBF"

FORCE_DOWNLOAD = True

MEDOID_TYPE = "dtw" # "dtw" or "euclidean"

STS_NUMBER = 40
STS_LENGTH = 10

TEST_SIZE = 0.3

BIN_NUMBER = 3
SAMPLE_SHFT = -10

RHO = 0.1

PLOTS = False
NPROCS = 4
RANDOM_STATE = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    # define data folder
    save_path = Path.cwd() / "data" / DATASET_NAME
    save_path.mkdir(parents=True, exist_ok=True)

    # SHARED DATA FOR ALL TASKS       
    #############################

    save_file = save_path / "DB_shared.npz"
    if FORCE_DOWNLOAD or not save_file.exists():

        # grab the data
        X, Y, mapping = acquire_dataset(DATASET_NAME)

        # train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, 
                                            random_state=RANDOM_STATE, stratify=Y, shuffle=True)

        # compute medoids 
        medoids, medoid_ids = compute_medoids(X_train, Y_train, distance_type=MEDOID_TYPE)

        # plot the medoids
        img_folder = save_path / "images" / "medoids"
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
            medoids=medoids, medoids_ids=medoid_ids,
            mapping=mapping, test_size=TEST_SIZE)
    
    else:

        with np.load(save_file) as data:
            X, Y, mapping = data["X"], data["Y"], data["mapping"] 
    

    # DATA FOR MAIN CLASSIFICATION TASK       
    ####################################
    log.info("Generating data for the main task")
    
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

    #  DATA FOR AUXILIARY CLASSIFICATION TASK       
    ###########################################
    log.info("Generating data for the aux task")

    save_file_train = save_path / "DB-train_aux-task.npz"
    save_file_test = save_path / "DB-test_aux-task.npz"
    

    # TODO check cross information
    # discretize each time series to obtain new labels
    STSs_Yd_train = discretize_STSs(STSs_X_train, intervals=BIN_NUMBER, strategy="quantile")
    STSs_Yd_test = discretize_STSs(STSs_X_test, intervals=BIN_NUMBER, strategy="quantile")

    # shift the labels
    STSs_Xs_train, STSs_Yds_train = shift_STSs_labels(STSs_X_train, STSs_Yd_train, SAMPLE_SHFT)
    STSs_Xs_test, STSs_Yds_test = shift_STSs_labels(STSs_X_test, STSs_Yd_test, SAMPLE_SHFT)

    # compute the ESMs
    ESMs_train = compute_OESMs(STSs_Xs_train, medoids, rho=RHO, nprocs=NPROCS)
    ESMs_test = compute_OESMs(STSs_Xs_test, medoids, rho=RHO, nprocs=NPROCS)

    # save the data
    np.savez_compressed(save_file_train, STSs=STSs_Xs_train, labels=STSs_Yds_train, ESMs=ESMs_train)
    np.savez_compressed(save_file_test, STSs=STSs_Xs_test, labels=STSs_Yds_test, ESMs=ESMs_test)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
