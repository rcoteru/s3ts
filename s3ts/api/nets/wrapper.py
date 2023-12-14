#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Wrapper model for the deep learning models. """

from __future__ import annotations
from typing import Literal

# modules
from pytorch_lightning import LightningModule
from s3ts.api.nets.encoders.frames.CNN import CNN_IMG
from s3ts.api.nets.encoders.frames.RES import RES_IMG
from s3ts.api.nets.encoders.series.RNN import RNN_TS
from s3ts.api.nets.encoders.series.CNN import CNN_TS
from s3ts.api.nets.encoders.series.RES import RES_TS
from s3ts.api.nets.decoders.linear import LinearDecoder

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
import numpy as np
import torch

encoder_dict = {"img": {"cnn": CNN_IMG, "res": RES_IMG},
    "ts": {"rnn": RNN_TS, "cnn": CNN_TS, "res": RES_TS}}

class WrapperModel(LightningModule):

    name: str           # model name
    dsrc: str           # input dsrc: ["img", "ts"]
    arch: str           # architecture: ["rnn", "cnn", "res"]
    task: str           # task type: ["cls", "reg"]

    wdw_len: int        # window length
    wdw_str: int        # window stride
    sts_str: bool       # stride the series too?

    n_dims: int         # number of STS dimensions
    n_classes: int      # number of classes
    n_patterns: int     # number of patterns
    l_patterns: int     # pattern size
    
    enc_feats: int      # encoder feature hyperparam
    dedsrcc_feats: int      # decoder feature hyperparam
    lr: float           # learning rate

    def __init__(self, dsrc, arch, task,
        n_dims, n_classes, n_patterns, l_patterns,
        wdw_len, wdw_str, sts_str,
        enc_feats, dec_feats, lr,
        name=None) -> None:

        """ Wrapper for the PyTorch models used in the experiments. """

        if name is None:
            name = f"{dsrc}_{arch}_{task}_wl{wdw_len}_ws{wdw_str}_ss{int(sts_str)}"
            if dsrc == "ts":
                name += f"_nd{n_dims}"
            elif dsrc == "img":
                name += f"_np{n_patterns}_lp{l_patterns}"
            if task == "cls":
                name += f"_n{n_classes}"

        # save parameters as attributes
        super().__init__(), self.__dict__.update(locals())
        self.save_hyperparameters()

        # select model architecture class
        enc_arch: LightningModule = encoder_dict[dsrc][arch]

        # create encoder
        if dsrc == "img":
            ref_size, channels = l_patterns, n_patterns
        elif dsrc == "ts":
            ref_size, channels = 1, self.n_dims
 
        encoder = enc_arch(channels=channels, ref_size=ref_size, 
            wdw_size=self.wdw_len, n_feature_maps=self.enc_feats)
        
        self.encoder = encoder

        # create decoder
        shape: torch.Tensor = self.encoder.get_output_shape()
        inp_feats = torch.prod(torch.tensor(shape[1:]))
        if self.task == "cls":
            out_feats = self.n_classes
        elif self.task == "reg":
            out_feats = self.wdw_len if self.sts_str else self.wdw_len*self.wdw_str
            out_feats = out_feats*n_dims
        self.decoder = LinearDecoder(inp_feats=inp_feats, 
            hid_feats=dec_feats, out_feats=out_feats, hid_layers=2)

        # create softmax and flatten layers
        self.flatten = nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax()

        # create metrics
        if self.task == "cls":
            for phase in ["train", "val", "test"]: 
                self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=out_feats, task="multiclass"))
                self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=out_feats, task="multiclass"))
                if phase != "train":
                    self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=out_feats, task="multiclass"))
        elif self.task == "reg":
            for phase in ["train", "val", "test"]:
                self.__setattr__(f"{phase}_mse", tm.MeanSquaredError(squared=False))
                self.__setattr__(f"{phase}_r2",  tm.R2Score(num_outputs=out_feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)
        if self.task == "cls":
            x = self.softmax(x)
        if self.task == "reg":
            bsize = x.shape[0]
            x = x.reshape((bsize, self.n_dims, -1))
        return x

    def _inner_step(self, batch: dict[str, torch.Tensor], stage: Literal["train", "val", "test"]) -> torch.Tensor:

        """ Inner step for the training, validation and testing. """

        # Forward pass
        if self.dsrc == "img":
            output: torch.Tensor = self(batch["frame"])
        elif self.dsrc == "ts":
            output: torch.Tensor = self(batch["series"])

        # Compute the loss and metrics
        if self.task == "cls":
            oh_label: torch.Tensor = F.one_hot(batch["label"], 
                                        num_classes=self.n_classes)
            loss = F.cross_entropy(output, oh_label.to(torch.float32)) # type: ignore
            acc = self.__getattr__(f"{stage}_acc")(output, batch["label"]) # type: ignore
            f1  = self.__getattr__(f"{stage}_f1")(output, batch["label"]) # type: ignore
            if stage != "train":
                auroc = self.__getattr__(f"{stage}_auroc")(output, batch["label"])   # type: ignore
        elif self.task == "reg":
            loss = F.mse_loss(output, batch["series"]) # type: ignore
            mse = self.__getattr__(f"{stage}_mse")(output, batch["series"]) # type: ignore
            r2 = self.__getattr__(f"{stage}_r2")(self.flatten(output), self.flatten(batch["series"])) # type: ignore

        # log loss and metrics
        on_step = True if stage == "train" else False
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=True, logger=True) # type: ignore
        if self.task == "cls":
            self.log(f"{stage}_acc", acc, on_epoch=True, on_step=False, prog_bar=True, logger=True) # type: ignore
            self.log(f"{stage}_f1", f1, on_epoch=True, on_step=False, prog_bar=False, logger=True) # type: ignore
            if stage != "train":
                self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False, prog_bar=True, logger=True) # type: ignore
        elif self.task == "reg":
            self.log(f"{stage}_mse", mse, on_epoch=True, on_step=False, prog_bar=True, logger=True) # type: ignore
            self.log(f"{stage}_r2", r2, on_epoch=True, on_step=False, prog_bar=True, logger=True) # type: ignore

        # return loss
        return loss.to(torch.float32) # type: ignore

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """ Validation step. """
        return self._inner_step(batch, stage="test")

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """ Predict step. """
        if self.dsrc == "img":
            output: torch.Tensor = self(batch["frame"])
        elif self.dsrc == "ts":
            output: torch.Tensor = self(batch["series"])
        return output

    def configure_optimizers(self):
        """ Configure the optimizers. """
        mode = "max" if self.task == "cls" else "min"
        monitor = "val_acc" if self.task == "cls" else "val_mse"
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 
                    mode=mode, factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": monitor,
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }