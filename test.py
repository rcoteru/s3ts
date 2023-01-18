# data
from s3ts.training.pretrain import pretrain_data_modules
from s3ts.frames.tasks.download import download_dataset
from s3ts.frames.base import BaseDataModule

# models
from s3ts.models.encoders.CNN import CNN_Encoder
from s3ts.models.base import BasicModel

# training
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

from pathlib import Path

# SETTINGS
# =================================

DIR = Path("test")

DATASET = "GunPoint"
PRETRAIN = 1

RANDOM_STATE = 0
seed_everything(RANDOM_STATE)

# DATA
# =================================
print("Loading data...")

X, Y, mapping = download_dataset(DATASET)

pretrain_dm, train_dm = pretrain_data_modules(
    X=X, Y=Y, 
    ulab_frac=0.8,
    test_size=0.2, 
    window_size=5,
    batch_size=128,
    rho_dfs=0.1,
    random_state=RANDOM_STATE
)

# PRETRAIN
# =================================

if PRETRAIN:
    print("Pretraining...")

    # create the model
    pretrain_dm: BaseDataModule
    pretrain_model = BasicModel(
            n_labels=pretrain_dm.n_labels, 
            n_patterns=pretrain_dm.n_patterns,
            l_patterns=pretrain_dm.l_patterns,
            window_size=pretrain_dm.window_size,
            arch=CNN_Encoder)
    
    # create the trainer
    checkpoint = ModelCheckpoint(monitor="val_acc", mode="max")               # save best model version
    trainer = Trainer(default_root_dir=DIR / "pretrain",
        logger = TensorBoardLogger(save_dir= DIR / "pretrain", name="logs"),    # progress logs
        callbacks=[
            EarlyStopping(monitor="val_acc", mode="max", patience=5),         # early stop the model
            LearningRateMonitor(logging_interval='step'),                       # learning rate logger
            checkpoint],
        max_epochs=100,  deterministic = True,
        log_every_n_steps=1, check_val_every_n_epoch=1)

    trainer.fit(pretrain_model, datamodule=pretrain_dm)

    # load the best one and grab the encoder
    pretrain_model.load_from_checkpoint(checkpoint.best_model_path)
    #model.load_from_checkpoint("test/pretrain/model.ckpt")
    pretrain_encoder = pretrain_model 

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
        arch=CNN_Encoder)

if PRETRAIN:
    train_model.encoder = pretrain_encoder
    
# create the trainer
trainer = Trainer(default_root_dir=DIR / "finetune",
    logger = TensorBoardLogger(save_dir= DIR / "finetune", name="logs"),    # progress logs
    callbacks=[
        EarlyStopping(monitor="val_auroc", mode="max", patience=5),     # early stop the model
        LearningRateMonitor(logging_interval='step'),                   # learning rate logger
        ModelCheckpoint(monitor="val_auroc", mode="max")                # save best model version
        ],
    max_epochs=100,  deterministic = True,
    log_every_n_steps=1, check_val_every_n_epoch=1
)

trainer.fit(train_model, datamodule=train_dm)