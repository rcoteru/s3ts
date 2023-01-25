#/usr/bin/python3

# data processing stuff
from s3ts.frames.tasks.compute import compute_medoids, compute_STS
from s3ts.frames.tasks.download import download_dataset
from s3ts.frames.tasks.oesm import compute_OESM
from sklearn.preprocessing import KBinsDiscretizer

# data modules
from sklearn.model_selection import train_test_split
from s3ts.frames.pred import PredDataModule

# models
from pytorch_lightning import LightningModule
from s3ts.models.pred import PredModel

# training stuff
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer, seed_everything

# general
from shutil import rmtree
from pathlib import Path
import pandas as pd
import numpy as np


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prepare_data_modules(
        dataset: str,
        # these can be changed without recalculating anything
        batch_size: int,
        window_size: int,
        lab_shifts: list[int],
        # therse are needed for frame creation and imply recalcs
        rho_dfs: float,
        pret_frac: float,
        test_frac: float,
        # ~~~~~~~~~~~~~~~~
        nframes_tra: int, 
        nframes_pre: int,
        nframes_test: int,
        # ~~~~~~~~~~~~~~~~
        seed_sts: int = 0,
        seed_label: int = 0,
        seed_test: int = 0,
        # ~~~~~~~~~~~~~~~~
        cache_dir: Path = Path("cache")
        ) -> tuple[PredDataModule, PredDataModule]:

    """ Prepares the data modules. """

    print(f"Downloading dataset {dataset}...")
    X, Y, mapping = download_dataset(dataset)

    # set splitting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # división entre train y test
    print(f"Splitting training and test sets (seed: {seed_test})")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
            test_size=test_frac, stratify=Y, random_state=seed_test, shuffle=True)

    # divide en labeled y unlabeled
    print(f"Splitting train and pretrain sets (seed: {seed_label})")
    X_pre, X_tra, Y_pre, Y_tra = train_test_split(X_train, Y_train, 
        test_size=pret_frac, stratify=Y_train, random_state=seed_label, shuffle=True)

    # pattern selection ~~~~~~~~~~~~~~~~~~~~~~~~~

    # selecciona los patrones [n_patterns,  l_patterns]
    print(f"Selecting the DFS patterns from the train data")
    medoids, medoid_ids = compute_medoids(X_tra, Y_tra, distance_type="dtw")

    # STS generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("Generating 'train' STS...")
    STS_tra, labels_tra = compute_STS(X=X_tra,Y=Y_tra, target_nframes=4000, 
        frame_buffer=window_size*3,random_state=seed_sts)

    print("Generating 'pretrain' STS...")
    STS_pre, _, = compute_STS(X=X_pre, Y=Y_pre, target_nframes=4000, 
        frame_buffer=window_size*3,random_state=seed_sts)
    kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile", random_state=seed_sts)
    kbd.fit(STS_pre.reshape(-1,1))
    labels_pre = kbd.transform(STS_pre.reshape(-1,1)).squeeze().astype(int)
    
    print("Generating 'test' STS...")
    STS_test, labels_test = compute_STS(X=X_test, Y=Y_test, target_nframes=4000, 
        frame_buffer=window_size*3,random_state=seed_sts)

    # DFS generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fracs = f"{test_frac}-{pret_frac}"
    seeds = f"{seed_test}-{seed_label}-{seed_sts}"
    frames = f"{nframes_tra}-{nframes_pre}-{nframes_test}"
    cache_file = cache_dir / f"DFS_{dataset}_{fracs}_{seeds}_{frames}.npz"

    if not Path(cache_file).exists():
        print("Generating 'train' DFS...")
        DFS_tra = compute_OESM(STS_tra, medoids, rho=rho_dfs)   # generate DFS
        print("Generating 'pretrain' DFS...")
        DFS_pre = compute_OESM(STS_pre, medoids, rho=rho_dfs) 
        print("Generating 'test' DFS...")
        DFS_test = compute_OESM(STS_test, medoids, rho=rho_dfs) 
        np.savez_compressed(cache_file, DFS_tra=DFS_tra, DFS_pre=DFS_pre, DFS_test=DFS_test)
    else:
        print("Loading DFS from cached file...")
        with np.load(cache_file) as data:
            DFS_tra, DFS_pre, DFS_test = data["DFS_tra"], data["DFS_pre"], data["DFS_test"]

    # create data modules ~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    print("Creating 'train' dataset...")

    # create data module (train)
    dm_tra = PredDataModule(
        STS_train=STS_tra, DFS_train=DFS_tra, labels_train=labels_tra, nframes_train=nframes_tra,
        STS_test=STS_test, DFS_test=DFS_test, labels_test=labels_test, nframes_test=nframes_test,
        window_size=window_size, batch_size=batch_size, lab_shifts=[0])

    print("Creating 'pretrain' dataset...")

    l_sample = X.shape[1]
    lab_shifts = np.round(np.array(lab_shifts)*l_sample).astype(int)
    print("Label shifts:", lab_shifts)    

    # create data module (pretrain)
    dm_pre = PredDataModule(
        STS_train=STS_pre, DFS_train=DFS_pre, labels_train=labels_pre, nframes_train=nframes_pre,
        window_size=window_size, batch_size=batch_size, lab_shifts=lab_shifts)   

    return dm_tra, dm_pre

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def setup_trainer(
    directory: Path,
    version: str,
    epoch_max: int,
    epoch_patience: int,
    stop_metric: str = "val_f1",
    ) -> tuple[Trainer, ModelCheckpoint]:

    checkpoint = ModelCheckpoint(monitor=stop_metric, mode="max")    
    trainer = Trainer(default_root_dir=directory,  accelerator="auto",
        # progress logs
        loggers = [
            TensorBoardLogger(save_dir=directory, name="logs",),
            CSVLogger(save_dir=directory, name="logs")
        ],
        callbacks=[
            # early stop the model
            EarlyStopping(monitor=stop_metric, mode="max", patience=epoch_patience),         
            LearningRateMonitor(logging_interval='step'),  # learning rate logger
            checkpoint  # save best model version
            ],
        max_epochs=epoch_max,  deterministic = False,
        log_every_n_steps=1, check_val_every_n_epoch=1
    )

    return trainer, checkpoint

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compare_pretrain(
    dataset: str,
    arch: type[LightningModule],
    # ~~~~~~~~~~~~~~~~
    batch_size: int,
    window_size: int,
    lab_shifts: list[int],
    # ~~~~~~~~~~~~~~~~
    # therse are needed for frame creation and imply recalcs
    rho_dfs: float,
    pret_frac: float,
    test_frac: float,
    # ~~~~~~~~~~~~~~~~
    nframes_tra: int, 
    nframes_pre: int,
    nframes_test: int,
    # ~~~~~~~~~~~~~~~~
    pre_patience: int = 5,
    pre_maxepoch: int = 100,
    tra_patience: int = 40,
    tra_maxepoch: int = 200,
    # ~~~~~~~~~~~~~~~~
    stop_metric: str = "val_f1",
    # ~~~~~~~~~~~~~~~~
    seed_sts: int = 0,
    seed_label: int = 0,
    seed_test: int = 0,
    seed_torch: int = 0
    ) -> pd.Series:

    seed_everything(seed_torch)

    results = pd.Series()
    results["dataset"], results["decoder"] = dataset, str(type(LightningModule))
    results["batch_size"], results["window_size"], results["lab_shifts"] = batch_size, window_size, lab_shifts
    results["nframes_tra"], results["nframes_pre"],results["nframes_test"] = nframes_tra, nframes_pre, nframes_test
    results["seed_sts"], results["seed_label"], results["seed_test"] = seed_sts, seed_label, seed_test 

    train_dm, pretrain_dm = prepare_data_modules(dataset=dataset,
        batch_size=batch_size, window_size=window_size, lab_shifts=lab_shifts,
        rho_dfs=rho_dfs, pret_frac=pret_frac, test_frac=test_frac,
        nframes_tra=nframes_tra, nframes_pre=nframes_pre, nframes_test=nframes_test,
        seed_sts=seed_sts, seed_label=seed_label, seed_test=seed_test)

    train_dm: PredDataModule
    pretrain_dm: PredDataModule

    # ~~~~~~~~~~~~~~~~~~~~~ train without pretrain

    # create the model
    train_model = PredModel(
            n_labels=train_dm.n_labels, 
            n_patterns=train_dm.n_patterns,
            l_patterns=train_dm.l_patterns,
            window_size=train_dm.window_size,
            arch=arch) 

    # train the model
    trainer, checkpoint = setup_trainer()
    trainer.fit(train_model, datamodule=train_dm)
    train_model = train_model.load_from_checkpoint(checkpoint.best_model_path)

    valid_results = trainer.validate(train_model, datamodule=train_dm)
    test_results = trainer.test(train_model, datamodule=train_dm)
    
    # log results
    results["default_val_acc"] = valid_results[0]["val_acc"]
    results["default_val_f1"] = valid_results[0]["val_f1"]
    results["default_val_auroc"] = valid_results[0]["val_auroc"]

    results["default_test_acc"] = test_results[0]["test_acc"]
    results["default_test_f1"] = test_results[0]["test_f1"]
    results["default_test_auroc"] = test_results[0]["test_auroc"]

    # ~~~~~~~~~~~~~~~~~~~~~ do the pretrain

    # create the model
    pretrain_model = PredModel(
            n_labels=pretrain_dm.n_labels, 
            n_patterns=pretrain_dm.n_patterns,
            l_patterns=pretrain_dm.l_patterns,
            window_size=pretrain_dm.window_size,
            lab_shifts=pretrain_dm.lab_shifts,
            arch=arch)
               
    trainer, checkpoint = setup_trainer()
    trainer.fit(train_model, datamodule=train_dm)
    pretrain_model = pretrain_model.load_from_checkpoint(checkpoint.best_model_path)
    valid_results = trainer.validate(train_model, datamodule=train_dm)

    # log results
    results["aux_val_acc"] = valid_results[0]["val_acc"]
    results["aux_val_f1"] = valid_results[0]["val_f1"]
    
    # grab the pretrained encoder
    pretrained_encoder = pretrain_model.encoder

    # ~~~~~~~~~~~~~~~~~~~~~ train with pretrain

    train_model = PredModel(
            n_labels=train_dm.n_labels, 
            n_patterns=train_dm.n_patterns,
            l_patterns=train_dm.l_patterns,
            window_size=train_dm.window_size,
            arch=arch) 
    train_model.encoder = pretrained_encoder

    trainer, checkpoint = setup_trainer(
        ...
    )
    trainer.fit(train_model, datamodule=train_dm)
    train_model = train_model.load_from_checkpoint(checkpoint.best_model_path)
    valid_results = trainer.validate(train_model, datamodule=train_dm)
    test_results = trainer.test(train_model, datamodule=train_dm)

    # log results
    results["pretrain_val_acc"] = valid_results[0]["val_acc"]
    results["pretrain_val_f1"] = valid_results[0]["val_f1"]
    results["pretrain_val_auroc"] = valid_results[0]["val_auroc"]

    results["pretrain_test_acc"] = test_results[0]["test_acc"]
    results["pretrain_test_f1"] = test_results[0]["test_f1"]
    results["pretrain_test_auroc"] = test_results[0]["test_auroc"]
