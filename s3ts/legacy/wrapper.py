#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Wrapper model for the S3TS models. """

from __future__ import annotations

# lightning
from s3ts.models.decoders.linear import LinearDecoder
from s3ts.models.decoders.lstm import LSTMDecoder
from pytorch_lightning import LightningModule

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
import torch

# architectures
from s3ts.models.encoders.frames.CNN import CNN_DF
from s3ts.models.encoders.frames.RES import RES_DF

from s3ts.models.encoders.series.RNN import RNN_TS
from s3ts.models.encoders.series.CNN import CNN_TS
from s3ts.models.encoders.series.RES import RES_TS

# numpy
import logging as log
import numpy as np

class WrapperModel(LightningModule):

    def __init__(self,
        mode: str, 
        arch: str,
        task: str,
        n_classes: int,
        n_patterns: int,
        l_patterns: int,
        window_length: int,
        window_time_stride: int,
        window_patt_stride: int,
        stride_series: bool,
        encoder_feats: int,
        decoder_feats: int,
        learning_rate: float,
        ):

        """ Wrapper for the PyTorch models used in the experiments. """

        super(WrapperModel, self).__init__()
        
        self.encoder_dict = {
            "ts": {"rnn": RNN_TS, "cnn": CNN_TS, "res": RES_TS}, 
            "df": {"cnn": CNN_DF, "res": RES_DF},
            "gf": {"cnn": CNN_DF, "res": RES_DF}}
        encoder_arch = self.encoder_dict[mode][arch]

        # Check decoder parameters
        if task not in ["cls", "reg"]:
            raise ValueError(f"Invalid task: {task}")

        # Gather model parameters
        self.mode = mode
        self.arch = arch
        self.target = task
        self.n_classes = n_classes
        self.n_patterns = n_patterns
        self.l_patterns = l_patterns
        self.window_length = window_length
        self.window_time_stride = window_time_stride
        self.window_patt_stride = window_patt_stride
        self.stride_series = stride_series
        self.encoder_feats = encoder_feats
        self.decoder_feats = decoder_feats
        self.learning_rate = learning_rate

        # Save hyperparameters
        self.save_hyperparameters({"mode": mode, "arch": arch, "target": target,
            "n_classes": n_classes, "n_patterns": n_patterns, "l_patterns": l_patterns,
            "window_length": window_length, "stride_series": stride_series,
            "window_time_stride": window_time_stride, "window_patt_stride": window_patt_stride,
            "encoder_feats": encoder_feats, "decoder_feats": decoder_feats, "learning_rate": learning_rate})
        
        # Create the encoder
        if mode == "df" or mode == "gf":
            ref_size = len(np.arange(self.l_patterns)[::self.window_patt_stride])
            channels = self.n_patterns
        elif mode == "ts":
            ref_size, channels = 1, 1 
        self.encoder = encoder_arch(channels=channels, ref_size=ref_size, wdw_size=window_length,
            n_feature_maps=encoder_feats)
        
        # Determine the input size of the decoder
        shape: torch.tensor = self.encoder.get_output_shape()
        inp_feats = torch.prod(torch.tensor(shape[1:]))
        self.flatten = nn.Flatten(start_dim=1)

        # Determine the number of decoder output features
        if self.target == "cls":
            out_feats = self.n_classes
        elif self.target == "reg":
            if self.stride_series:
                out_feats = self.window_length 
            else:
                out_feats = self.window_length*self.window_time_stride

        # Create the decoder
        self.decoder = LinearDecoder(
            inp_feats=inp_feats,
            hid_feats=decoder_feats,
            out_feats=out_feats,
            hid_layers=2,
        )

        # Create the softmax layer
        self.softmax = nn.Softmax()

        # Add the metrics
        if self.target == "cls":
            for phase in ["train", "val", "test"]: 
                self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=out_feats, task="multiclass"))
                self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=out_feats, task="multiclass"))
                if phase != "train":
                    self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=out_feats, task="multiclass"))
        elif self.target == "reg":
            for phase in ["train", "val", "test"]:
                self.__setattr__(f"{phase}_mse", tm.MeanSquaredError(squared=False))
                self.__setattr__(f"{phase}_r2",  tm.R2Score(num_outputs=out_feats))


    def forward(self, x: torch.Tensor):
        """ Forward pass. """
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)
        if self.target == "cls":
            x = self.softmax(x)
        return x

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _inner_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
            stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Unpack the batch from the dataloader
        # frames, series, label = batch

        # Forward pass
        if self.mode == "df" or self.mode == "gf":
            output = self(batch[0])
        elif self.mode == "ts":
            output = self(torch.unsqueeze(batch[1] , dim=1))

        # Compute the loss and metrics
        if self.target == "cls":
            loss = F.cross_entropy(output, batch[2].to(torch.float32))
            acc = self.__getattr__(f"{stage}_acc")(output, torch.argmax(batch[2], dim=1))
            f1  = self.__getattr__(f"{stage}_f1")(output, torch.argmax(batch[2], dim=1))
            if stage != "train":
                auroc = self.__getattr__(f"{stage}_auroc")(output, torch.argmax(batch[2], dim=1))  
        elif self.target == "reg":
            loss = F.mse_loss(output, batch[1])
            mse = self.__getattr__(f"{stage}_mse")(output,  batch[1])
            r2 = self.__getattr__(f"{stage}_r2")(output,  batch[1])

        # Log the loss and metrics
        on_step = True if stage == "train" else False
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)
        if self.target == "cls":
            self.log(f"{stage}_acc", acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log(f"{stage}_f1", f1, on_epoch=True, on_step=False, prog_bar=False, logger=True)
            if stage != "train":
                self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        elif self.target == "reg":
            self.log(f"{stage}_mse", mse, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log(f"{stage}_r2", r2, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        # Return the loss
        return loss.to(torch.float32)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """ Test step. """
        return self._inner_step(batch, stage="test")

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def configure_optimizers(self):

        """ Configure the optimizers. """

        if self.target == "cls":
            mode, monitor = "max", "val_acc"
        elif self.target == "reg":
            mode, monitor = "min", "val_mse"

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode=mode, factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": monitor,
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }