# data
from s3ts.frames.tasks.compute import compute_medoids, compute_STS
from s3ts.frames.tasks.download import download_dataset
from s3ts.frames.tasks.oesm import compute_OESM
from s3ts.frames.base import BaseDataModule

# models
from s3ts.models.encoders.CNN import CNN_Encoder
from s3ts.models.base import BasicModel

# training
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from pathlib import Path
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

DIR = Path("test")

DATASET = "GunPoint"
ULAB_FRAC = 0.8
TEST_FRAC = 0.2

WINDOW_SIZE = 5
BATCH_SIZE = 32

RHO_DF = 0.1
RANDOM_STATE = 0

# ~~~~~~~~~~~~~~~~~~~~~~~ descarga el dataset
X, Y, mapping = download_dataset(DATASET)

# divide en labeled y unlabeled
X_lab, X_ulab, Y_lab, Y_ulab = train_test_split(X, Y, 
    test_size=ULAB_FRAC, stratify=Y, random_state=RANDOM_STATE, shuffle=True)

# labeled train test split
X_train, X_test, Y_train, Y_test = train_test_split(X_lab, Y_lab, 
    test_size=TEST_FRAC, stratify=Y_lab, random_state=RANDOM_STATE,  shuffle=True)

# ~~~~~~~~~~~~~~~~~~~~~~~ train dataset with lab

# labeled train test split
X_train, X_test, Y_train, Y_test = train_test_split(X_lab, Y_lab, 
    test_size=TEST_FRAC, stratify=Y_lab, random_state=RANDOM_STATE,  shuffle=True)

# selecciona los patrones [n_patterns,  l_patterns]
medoids, medoid_ids = compute_medoids(X_train, Y_train, distance_type="dtw")

file_lab = "cache/lab.npy"
STS_lab, labels_lab = compute_STS(X_lab, Y_lab)            # generate STS
if not Path(file_lab).exists(): 
    DFS_lab = compute_OESM(STS_lab, medoids, rho=RHO_DF)   # generate DFS
    np.save(file_lab, DFS_lab)
else:
    DFS_lab = np.load(file_lab)

# ~~~~~~~~~~~~~~~~~~~~~~~ pretrain dataset with ulab

file_ulab = "cache/ulab.npy"
STS_ulab, _ = compute_STS(X_ulab, Y_ulab)                   # generate STS
if not Path(file_ulab).exists(): 
    DFS_ulab = compute_OESM(STS_ulab, medoids, rho=RHO_DF)  # generate DFS
    np.save(file_ulab, DFS_ulab)
else:
    DFS_ulab = np.load(file_ulab)


# generate labels
kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile", random_state=RANDOM_STATE)
kbd.fit(STS_ulab.reshape(-1,1))
labels_ulab = kbd.transform(STS_ulab.reshape(-1,1)).squeeze().astype(int)

# ~~~~~~~~~~~~~~~~~~~~~~~ pretrain

# create the data module
dm = BaseDataModule(STS=STS_ulab, labels=labels_ulab, DFS=DFS_ulab, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)

# create the model
model = BasicModel(
        n_labels=dm.n_labels, 
        n_patterns=dm.n_patterns,
        l_patterns=dm.l_patterns,
        window_size=dm.window_size,
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
#trainer.fit(model, datamodule=dm)

# load the best one and grab the encoder
#model.load_from_checkpoint(checkpoint.best_model_path)
model.load_from_checkpoint("test/pretrain/model.ckpt")
encoder = model.encoder

# ~~~~~~~~~~~~~~~~~~~~~~~ train

# create the data module
dm = BaseDataModule(STS=STS_lab, labels=labels_lab, DFS=DFS_lab, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)
 
# create the model
model = BasicModel(
        n_labels=dm.n_labels, 
        n_patterns=dm.n_patterns,
        l_patterns=dm.l_patterns,
        window_size=dm.window_size,
        arch=CNN_Encoder)
#model.encoder = encoder
    
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

trainer.fit(model, datamodule=dm)
encoder = model.encoder