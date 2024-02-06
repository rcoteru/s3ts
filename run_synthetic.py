#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Automatic training script for the paper's experiments. """

from s3ts.exp.settings import ExperimentSettings, SlurmSettings
from s3ts.exp.loop import run_loop
from typing import Literal
from pathlib import Path

# Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Choose the experiments to run
CTABLE_PRET = 0         # Pretrain the DF encoders for the table comparison
CTABLE_RUN  = 0         # Train loop for the table comparison (DL METHODS)

ABLAT_PRET = 0          # Pretrain the DF encoders (ablation study)
ABLAT_TIMEDIL = 0       # Ablation Study: Time dilation
ABLAT_SELFSUP = 0       # Ablation Study; Self-supervised pretraining

# SLURM settings
USE_SBATCH = 0
SETTINGS_YAML = Path("slurm/hipat-large.yaml")

# Datasets
DATASETS: dict[str, int] = {
    # dset name, target window size 
    "ArrowHead": 120,
    "CBF": 60,
    "ECG200": 50,
    "GunPoint": 70,
    "SyntheticControl": 30,
    "Trace": 150,                                        
}

# Architectures
ARCHS: dict[Literal["ts", "df", "gf"], dict[Literal["nn", "rnn", "cnn", "res"],int]] = {
    # data source: {architecture: encoder features}
    "ts": {"nn": 0, "rnn": 40, "cnn": 48, "res": 16},
    "df": {"cnn": 20, "res": 12},
    "gf": {"cnn": 20, "res": 12}}
DEC_FEATS: int = 64

# Model parameters
WDW_LEN = [10]
WDW_STR = [1,3,5,7]

# Data parameters
RHO_DFS: float = 0.1                # Memory parameter for DF
NUM_MED: list[int] = [1,2,3,4]
NSAMP_TRA: int = 1000
NSAMP_TST: int = 1000
NSAMP_PRE: int = 1000
VAL_SIZE: float = 0.25              # Validation size
CV_REPS: list[int] = list(range(5)) 
RANDOM_SEED = 0

# Training parameters
BATCH_SIZE: int = 128               # Batch size
MAX_EPOCHS_PRE: int = 60            # Max pretraining epochs
MAX_EPOCHS_TRA: int = 120           # Max training epochs
LEARNING_RATE: float = 1E-4         # Learning rate

# Directories
DIR_RESULTS = Path("results/")
DIR_ENCODERS = Path("encoders/")
DIR_DATASETS = Path("datasets/")
DIR_TRAINING = Path("training/experiments")

# Shared experiment setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Load SLURM settings if needed
SLURM_SETTINGS = None
if USE_SBATCH:
    SLURM_SETTINGS = SlurmSettings.load_from_yaml(SETTINGS_YAML)

# Setup shared settings
SHARED_ARGS = {"rho_dfs": RHO_DFS, "batch_size": BATCH_SIZE, "val_size": VAL_SIZE,
    "dec_feats": DEC_FEATS,  "lr": LEARNING_RATE, "seed": RANDOM_SEED,
    "dir_results": DIR_RESULTS,"dir_encoders": DIR_ENCODERS,
    "dir_datasets": DIR_DATASETS, "dir_training": DIR_TRAINING}

# Experiment loop for the comparison table
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Pretrain encoders for the comparison table
if CTABLE_PRET:
    dsrcs: list[Literal["df","gf"]] = ["df", "gf"]
    for dsrc in dsrcs:
        for arch in ARCHS[dsrc]:
            for dset in DATASETS:

                # Window parameters
                wdw_len = WDW_LEN[0]
                wdw_str = DATASETS[dset]//WDW_LEN[0]

                # Full series
                setts = ExperimentSettings(
                    dset=dset, dsrc=dsrc, arch=arch,
                    wdw_len=wdw_len, wdw_str=wdw_str,
                    enc_feats=ARCHS[dsrc][arch],
                    str_sts=False,
                    **SHARED_ARGS)
                run_loop(setts, slurm_settings=SLURM_SETTINGS)
                # Strided series
                setts = ExperimentSettings(
                    dset=dset, dsrc=dsrc, arch=arch,
                    wdw_len=wdw_len, wdw_str=wdw_str,
                    enc_feats=ARCHS[dsrc][arch],
                    str_sts=True,
                    **SHARED_ARGS)
                run_loop(setts, slurm_settings=SLURM_SETTINGS)


# Run experiments for the comparison table
if CTABLE_RUN:
    for cv_rep in CV_REPS:
        
        # TS methods
        dsrcs: list[Literal["ts"]] = ["ts"]
        for dsrc in dsrcs:
            for arch in ARCHS[dsrc]:
                for dset in DATASETS:
                    wdw_len = -1 if arch == "nn" else DATASETS[dset]
                    setts = ExperimentSettings(
                        dset=dset, dsrc=dsrc, arch=arch,
                        wdw_len=wdw_len, str_sts=False, wdw_str=1,
                        enc_feats=ARCHS[dsrc][arch],
                        **SHARED_ARGS)
                    run_loop(setts, slurm_settings=SLURM_SETTINGS)
                
        # IMG methods
        dsrcs: list[Literal["df","gf"]] = ["df", "gf"]
        for dsrc in dsrcs:
            for arch in ARCHS[dsrc]:
                for dset in DATASETS:

                    # Window parameters
                    wdw_len = WDW_LEN[0]
                    wdw_str = DATASETS[dset]//WDW_LEN[0]

                    # No pretrain
                    setts = ExperimentSettings(
                    dset=dset, dsrc=dsrc, arch=arch,
                    wdw_len=wdw_len, wdw_str=wdw_str,
                    enc_feats=ARCHS[dsrc][arch],
                    str_sts=False,
                    **SHARED_ARGS)
                    run_loop(setts, slurm_settings=SLURM_SETTINGS)

                    # Full series pretrain
                    setts = ExperimentSettings(
                    dset=dset, dsrc=dsrc, arch=arch,
                    wdw_len=wdw_len, wdw_str=wdw_str,
                    enc_feats=ARCHS[dsrc][arch],
                    str_sts=False,
                    **SHARED_ARGS)
                    run_loop(setts, slurm_settings=SLURM_SETTINGS)

                    # Strided series pretrain
                    setts = ExperimentSettings(
                    dset=dset, dsrc=dsrc, arch=arch,
                    wdw_len=wdw_len, wdw_str=wdw_str,
                    enc_feats=ARCHS[dsrc][arch],
                    str_sts=True,
                    **SHARED_ARGS)
                    run_loop(setts, slurm_settings=SLURM_SETTINGS)

# Experiment loop for the ablation study
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Ablation study: time dilation
if ABLAT_TIMEDIL:
    for cv_rep in CV_REPS:
        # IMG methods
        dsrcs: list[Literal["df","gf"]] = ["df", "gf"]
        for dsrc in dsrcs:
            for arch in ARCHS[dsrc]:
                for dset in DATASETS:
                    # Window parameters
                    wdw_len = WDW_LEN[0]
                    for wdw_str in WDW_STR:
                        # No pretrain
                        setts = ExperimentSettings(
                        dset=dset, dsrc=dsrc, arch=arch,
                        wdw_len=wdw_len, wdw_str=wdw_str,
                        enc_feats=ARCHS[dsrc][arch],
                        str_sts=False,
                        **SHARED_ARGS)
                        run_loop(setts, slurm_settings=SLURM_SETTINGS)

# Pretrain encoders for the ablation study
if ABLAT_PRET:
    dsrcs: list[Literal["df","gf"]] = ["df", "gf"]
    for dsrc in dsrcs:
        for arch in ARCHS[dsrc]:
            for dset in DATASETS:
                # Window parameters
                wdw_len = WDW_LEN[0]
                for wdw_str in WDW_STR:
                    # Full series
                    setts = ExperimentSettings(
                        dset=dset, dsrc=dsrc, arch=arch,
                        wdw_len=wdw_len, wdw_str=wdw_str,
                        enc_feats=ARCHS[dsrc][arch],
                        str_sts=False,
                        **SHARED_ARGS)
                    run_loop(setts, slurm_settings=SLURM_SETTINGS)
                    # Strided series
                    setts = ExperimentSettings(
                        dset=dset, dsrc=dsrc, arch=arch,
                        wdw_len=wdw_len, wdw_str=wdw_str,
                        enc_feats=ARCHS[dsrc][arch],
                        str_sts=True,
                        **SHARED_ARGS)
                    run_loop(setts, slurm_settings=SLURM_SETTINGS)


