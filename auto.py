"""
Kind obvious tbh.

@author Raúl Coterillo
@version 2023-01
"""

from s3ts.frames.tasks.download import download_dataset

# data
from s3ts.models.encoders.ResNet import ResNet_Encoder
from s3ts.models.encoders.CNN import CNN_Encoder
from s3ts.setup import compare_pretrain

from sklearn.model_selection import StratifiedKFold

from datetime import datetime
from itertools import product
from pathlib import Path
import pandas as pd
import numpy as np
import torch

torch.set_float32_matmul_precision("medium")

# SETTINGS
# =================================

DIR = Path("training/auto")

SEED = 0
NSPLITS  = 10

RHO_DFS = 0.1
WINDOW_SIZE = 5
BATCH_SIZE = 128
PRETRAIN_FRAC = 0.8
DATASET = "GunPoint"

N_FRAMES_TRAIN = 5000
N_FRAMES_PRE = 5000
N_FRAMES_TEST = 5000

STOP_METRIC = "val_f1"
PRETRAIN_PATIENCE: int = 5
PRETRAIN_MAXEPOCH: int = 5
TRAIN_PATIENCE: int = 40
TRAIN_MAXEPOCH: int = 5

PRE_INTERVALS = [3, 5]
LAB_SHIFTS = [[0], [0.15], [0.3]]
ENCODERS = [CNN_Encoder, ResNet_Encoder]

# =================================

X, Y, mapping = download_dataset(DATASET)

runs = list()
print(f"Train-test K-Fold validation: ({NSPLITS} splits)")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):

    print(f"Fold {i}:")
    X_train, Y_train = X[train_index,:], Y[train_index]
    X_test, Y_test = X[test_index,:], Y[test_index]

    for j, (arch, lab_shifts, pre_intervals) in enumerate(
        product(ENCODERS, LAB_SHIFTS, PRE_INTERVALS)):

        date_flag = datetime.now().strftime("%Y-%m-%d_%H-%M")
        subdir_name = f"{i}_{j}_{date_flag}"

        run_data = compare_pretrain(
            dataset=DATASET, arch=arch, rho_dfs=RHO_DFS, lab_shifts=lab_shifts,
            X_train= X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
            directory=DIR / subdir_name, fold_number=i, pre_intervals=pre_intervals,
            batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
            pret_frac=PRETRAIN_FRAC, stop_metric=STOP_METRIC,
            nframes_tra=N_FRAMES_TRAIN, nframes_pre=N_FRAMES_PRE, nframes_test=N_FRAMES_TEST,
            pre_patience=PRETRAIN_PATIENCE, pre_maxepoch=PRETRAIN_MAXEPOCH,
            tra_patience=TRAIN_PATIENCE, tra_maxepoch=TRAIN_MAXEPOCH,
        )

        runs.append(run_data)
        pd.concat(runs, ignore_index=True).to_csv("results/results.csv", index=False)

