"""
Kind obvious tbh.

@author Raúl Coterillo
@version 2023-01
"""

from s3ts.frames.tasks.download import download_dataset

# data
from s3ts.models.encoders.ResNet import ResNet_Encoder
from s3ts.models.encoders.CNN import CNN_Encoder
from s3ts.setup.pred import compare_pretrain

from sklearn.model_selection import StratifiedKFold

from itertools import product
from pathlib import Path
import pandas as pd
import numpy as np

# SETTINGS
# =================================

DIR = Path("training")

SEED = 0
NSPLITS  = 10

RHO_DFS = 0.1
WINDOW_SIZE = 5
BATCH_SIZE = 128
PRETRAIN_FRAC = 0.8
DATASET = "GunPoint"

LAB_SHIFTS = [[0], [0.15], [0.3]]

STOP_METRIC = "val_f1"
PRETRAIN_PATIENCE: int = 5,
PRETRAIN_MAXEPOCH: int = 100,
TRAIN_PATIENCE: int = 40,
TRAIN_MAXEPOCH: int = 200,

ENCODERS = [CNN_Encoder]#, ResNet_Encoder]

# =================================


X, Y, mapping = download_dataset(DATASET)

runs = list()

print(f"Train-test K-Fold validation: ({NSPLITS} splits)")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):

    print(f"Fold {i}:")
    X_train, Y_train = X[train_index,:], Y[train_index]
    X_test, Y_test = X[test_index,:], Y[test_index]

    for j, (arch, lab_shifts) in enumerate(product(ENCODERS, LAB_SHIFTS)):

        run_data = compare_pretrain(
            dataset=DATASET, arch=arch, rho_dfs=RHO_DFS
            X_train= X_train, X_test=X_test, Y_train=Y_train,
            directory=DIR / "whatever shit i come up with", 
            batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
            pret_frac=PRETRAIN_FRAC, stop_metric=STOP_METRIC
            nframes_tra=N_FRAMES,
            pre_patience=PRETRAIN_PATIENCE, pre_maxepoch=PRETRAIN_MAXEPOCH,
            tra_patience=TRAIN_PATIENCE, tra_maxepoch=TRAIN_MAXEPOCH,
        )
        runs.append(run_data)
        pd.concat(runs, ignore_index=True).to_csv("results.csv")


# data = pd.DataFrame()

# for seed in SEEDS:
#     for seed
