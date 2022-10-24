"""

@author Raúl Coterillo
"""

from s3ts.network.architecture import model_wrapper, CNN_DTW
from s3ts import tasks
import s3ts

import logging
import sys

log = logging.Logger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s')

# Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

STS_NUMBER = 40
STS_LENGTH = 10

TEST_SIZE = 0.3
RHO = 0.1

PLOTS = False
NPROCS = 4

s3ts.RANDOM_STATE = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    # create experiment folder
    exp_path = tasks.data.prepare_experiment("prueba")

    # download dataset
    tasks.data.prepare_dataset(
        exp_path = exp_path,
        dataset = "GunPoint",
        test_size = 0.3,
        force = False)

    # prepare classification data for main task
    dm_main = tasks.data.prepare_classification_data(
        exp_path = exp_path,
        task_name = "main",
        sts_length = 10,
        label_type = "original",
        label_shft = 0,
        batch_size = 128,
        wndow_size = 5,
        force = False)

    # prepare classification data for aux task
    dm_aux = tasks.data.prepare_classification_data(
        exp_path = exp_path,
        task_name = "aux",
        sts_length = 200,
        label_type = "discrete_STS",
        label_shft = 0,
        batch_size = 128,
        wndow_size = 5,
        force = False)

    # create the model
    model = model_wrapper(
        model_architecture=CNN_DTW,
        ref_size=dm_main.ds_train.ESMs.shape[2],
        channels=dm_main.channels,
        labels=dm_main.labels_size,
        window_size=5,
        lr=1e-5
    )

    # run the sequence
    tasks.train.run_sequence(
        exp_path = exp_path,
        seq_name = "pretrain",
        main_task = ("main", dm_main)
        aux_tasks = [("aux", dm_aux)]
        model = model
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
