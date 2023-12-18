
#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Common functions for the experiments. """

# pl imports
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

# in-package imports
from s3ts.api.nets.wrapper import WrapperModel
from s3ts.data.base import StreamingFramesDM

# other imports
import numpy as np
import torch

# default pl settings
default_pl_kwargs: dict = {
    "default_root_dir": "training",
    "accelerator": "auto",
    "seed": 42
}

# default learning rate
default_lr = 1E-4

# default networks sizes
default_dec_feats: int = 64
default_enc_feats: dict[dict[int]] = { 
    "ts": {"rnn": 40, "cnn": 48, "res": 16},
    "img": {"cnn": 20, "res": 12, "simplecnn": 32}}

# metrics settings
metric_settings: dict = {
    "reg": {"all": ["mse", "r2"], "target": "val_mse", "mode": "min"},
    "cls": {"all": ["re", "f1", "auroc"], "target": "val_re", "mode": "max"}
}

default_voting = {"n": 1, "w": 1}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def create_model_from_DM(dm:  StreamingFramesDM, dsrc: str, arch: str, dec_arch: str, task: str, name: str = None,
        enc_feats: int = None, dec_feats: int = None, dec_layers: int = None, lr: float = default_lr, voting: dict = default_voting
        ) -> WrapperModel:
    
    # use defaults values if needed
    if enc_feats is None:
        enc_feats = default_enc_feats[dsrc][arch]
    if dec_feats is None:
        dec_feats = default_dec_feats
    
    # return the model
    return WrapperModel(
        name=name,
        dsrc=dsrc,
        arch=arch,
        dec_arch=dec_arch,
        task= task,
        wdw_len=dm.wdw_len,
        wdw_str=dm.wdw_str,
        sts_str=dm.sts_str,
        n_dims=dm.n_dims,
        n_classes=dm.n_classes,
        n_patterns=dm.n_patterns,
        l_patterns=dm.l_patterns,
        enc_feats=enc_feats,
        dec_feats=dec_feats,
        dec_layers=dec_layers,
        lr=lr,
        voting=voting)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def train_model(
        dm: StreamingFramesDM, 
        model: WrapperModel,
        max_epochs: int,
        pl_kwargs: dict = default_pl_kwargs,
        ) -> tuple[WrapperModel, dict]:
    
    # reset the random seed
    seed_everything(pl_kwargs["seed"], workers=True)

    # choose metrics
    metrics = metric_settings[model.task]

    # set up the trainer
    ckpt = ModelCheckpoint(monitor=metrics['target'], mode=metrics["mode"])    
    tr = Trainer(default_root_dir=pl_kwargs["default_root_dir"], 
    accelerator=pl_kwargs["accelerator"], callbacks=[ckpt], max_epochs=max_epochs,
    logger=TensorBoardLogger(save_dir=pl_kwargs["default_root_dir"], name=model.name))

    # train the model
    tr.fit(model=model, datamodule=dm)

    # load the best weights
    model = model.load_from_checkpoint(ckpt.best_model_path)

    # run the validation with the final weights
    data = tr.validate(model, datamodule=dm)

    return model, {f"val_{m}": data[0][f"val_{m}"] for m in metrics["all"]}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def test_model(dm: StreamingFramesDM, model: WrapperModel,
        pl_kwargs: dict = default_pl_kwargs) -> dict:
    
    # choose metrics
    metrics = metric_settings[model.task]

    # set up the trainer   
    tr = Trainer(default_root_dir=pl_kwargs["default_root_dir"],  
        accelerator=pl_kwargs["accelerator"], logger=[])
    
    # test the model
    data = tr.test(model, datamodule=dm)

    return {f"test_{m}": data[0][f"test_{m}"] for m in metrics["all"]}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def get_test_preds(dm: StreamingFramesDM, model: WrapperModel,
        pl_kwargs: dict = default_pl_kwargs) -> tuple[np.ndarray, np.ndarray]:
    
    # choose metrics
    metrics = metric_settings[model.task]

    # set up the trainer   
    tr = Trainer(default_root_dir=pl_kwargs["default_root_dir"],  
        accelerator=pl_kwargs["accelerator"], logger=[])
    
    # test the model
    preds = tr.predict(model, datamodule=dm, return_predictions=True)
    preds = torch.stack([pred for batch in preds for pred in batch]).numpy()

    # grab test data samples as numpy array
    data = []
    if model.task == "reg":
        for i in range(len(dm.ds_test)):
            data.append(dm.ds_test[i]["series"].numpy())
    if model.task == "cls":
        for i in range(len(dm.ds_test)):
            data.append(dm.ds_test[i]["label"].numpy())
    data = np.array(data)

    return data, preds

