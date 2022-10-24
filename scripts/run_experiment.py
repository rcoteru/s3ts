"""

@author Raúl Coterillo
"""


from s3ts import tasks

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

    exp_path = tasks.data.prepare_experiment("prueba1")

    tasks.data.prepare_dataset(
        exp_path = exp_path,
        dataset = "GunPoint",
        test_size = 0.3,
        force = False)

    dm_main = tasks.data.prepare_classification_data(
        exp_path = exp_path,
        task_name = "main",
        sts_length = 10,
        label_type = "original",
        label_shft = 0,
        batch_size = 128,
        wndow_size = 5,
        force = False)

    dm_aux = tasks.data.prepare_classification_data(
        exp_path = exp_path,
        task_name = "aux",
        sts_length = 200,
        label_type = "discrete_STS",
        label_shft = 0,
        batch_size = 128,
        wndow_size = 5,
        force = False)

    dm_

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
