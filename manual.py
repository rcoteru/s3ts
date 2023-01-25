"""
Kind obvious tbh.

@author Raúl Coterillo
@version 2023-01
"""

# data

from s3ts.setup.pred import prepare_data_modules
from s3ts.frames.pred import PredDataModule
from s3ts.frames.base import BaseDataModule

# models
from s3ts.models.encoders.ResNet import ResNet_Encoder
from s3ts.models.encoders.CNN import CNN_Encoder
from s3ts.models.pred import PredModel
from s3ts.models.base import BasicModel

# training
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

from shutil import rmtree
from pathlib import Path

# SETTINGS
# =================================

DIR = Path("test")

DATASET = "GunPoint"
PRETRAIN = 1
ENCODER = CNN_Encoder
LAB_SHIFTS = [0]

RANDOM_STATE = 0
RANDOM_STATE_TEST = 0
seed_everything(RANDOM_STATE)

# remove training files
rmtree(DIR, ignore_errors=True)

# DATA
# =================================
print("Loading data...")

train_dm, pretrain_dm = prepare_data_modules(
    dataset=DATASET,
    # these can be changed without recalculating anything
    batch_size=128,
    window_size=5,
    lab_shifts=LAB_SHIFTS,
    # these are needed for frame creation and imply recalcs
    rho_dfs = 0.1,
    pret_frac = 0.8,
    test_frac = 0.2,
    # ~~~~~~~~~~~~~~~~
    nframes_tra = 2000, 
    nframes_pre = 2000,
    nframes_test = 2000,
    # ~~~~~~~~~~~~~~~~
    seed_sts=0,
    seed_label=0,
    seed_test=0,
    # ~~~~~~~~~~~~~~~~
)

# PRETRAIN
# =================================

if PRETRAIN:
    print("Pretraining...")

    # create the model
    pretrain_dm: PredDataModule
    pretrain_model = PredModel(
            n_labels=pretrain_dm.n_labels, 
            n_patterns=pretrain_dm.n_patterns,
            l_patterns=pretrain_dm.l_patterns,
            window_size=pretrain_dm.window_size,
            lab_shifts=pretrain_dm.lab_shifts,
            arch=ENCODER)
    
    # create the trainer
    checkpoint = ModelCheckpoint(monitor="val_f1", mode="max")                 # save best model version
    trainer = Trainer(default_root_dir=DIR / "pretrain",  accelerator="auto",
        logger = TensorBoardLogger(save_dir= DIR / "pretrain", name="logs"),    # progress logs
        callbacks=[
            EarlyStopping(monitor="val_f1", mode="max", patience=5),           # early stop the model
            LearningRateMonitor(logging_interval='step'),                       # learning rate logger
            checkpoint],
        max_epochs=5,  deterministic = False,
        log_every_n_steps=1, check_val_every_n_epoch=1)

    trainer.fit(pretrain_model, datamodule=pretrain_dm)

    # load the best one and grab the encoder
    pretrain_model = pretrain_model.load_from_checkpoint(checkpoint.best_model_path)

    trainer.validate(pretrain_model, datamodule=pretrain_dm)
    trainer.test(pretrain_model, datamodule=pretrain_dm)

    # x = trainer.validate(pretrain_model, datamodule=pretrain_dm)
    # x[0]['val_acc'], x[0]['val_f1']

    pretrain_encoder = pretrain_model.encoder

# TRAIN
# =================================
print("Training...")

# create the model
train_dm: BaseDataModule
train_model = BasicModel(
        n_labels=train_dm.n_labels, 
        n_patterns=train_dm.n_patterns,
        l_patterns=train_dm.l_patterns,
        window_size=train_dm.window_size,
        arch=ENCODER)

if PRETRAIN:
    train_model.encoder = pretrain_encoder
    
# create the trainer
checkpoint = ModelCheckpoint(monitor="val_acc", mode="max")               
trainer = Trainer(default_root_dir=DIR / "finetune",  accelerator="auto",
    # progress logs
    logger = TensorBoardLogger(save_dir= DIR / "finetune", name="logs"),    
    callbacks=[
        # early stop the model
        EarlyStopping(monitor="val_acc", mode="max", patience=40),         
        # learning rate logger
        LearningRateMonitor(logging_interval='step'),  
        # save best model version                     
        checkpoint                
        ],
    max_epochs=200,  deterministic = False,
    log_every_n_steps=1, check_val_every_n_epoch=1
)

trainer.fit(train_model, datamodule=train_dm)

# load the best model
train_model = train_model.load_from_checkpoint(checkpoint.best_model_path)

trainer.validate(train_model, datamodule=train_dm)
trainer.test(train_model, datamodule=train_dm)