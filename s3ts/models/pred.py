"""
Base Convolutional Classification Model

@author Raúl Coterillo
@version 2023-01
"""

from __future__ import annotations

# lightning
from pytorch_lightning import LightningModule

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
import torch

# numpy
import numpy as np
import logging

log = logging.Logger(__name__)

# ========================================================= #
#                     MULTITASK MODEL                       #
# ========================================================= #

class PredModel(LightningModule):

    def __init__(self,      
        n_labels: int,
        n_patterns: int, 
        l_patterns: int,
        window_size: int,
        lab_shifts: list[int],
        arch: type[LightningModule],
        learning_rate: float = 1e-4
        ):

        super().__init__()
        self.save_hyperparameters()

        self.n_labels = n_labels
        self.n_patterns = n_patterns
        self.l_patterns = l_patterns
        self.window_size = window_size
        self.learning_rate = learning_rate

        # encoder
        self.encoder: LightningModule = arch(
            ref_size=l_patterns, 
            channels=n_patterns, 
            window_size=window_size)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.encoder.get_output_shape(), 
            out_features=n_labels*len(lab_shifts)), nn.Softmax())

        # configure loggers
        for phase in ["train", "val", "test"]:
            self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=n_labels, task="multilabel"))
            self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=n_labels, task="multilabel", average="micro"))

    # FORWARD
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def forward(self, frame):

        """ Use for inference only (separate from training_step)"""

        linear = self.decoder(self.encoder(frame))
        out = torch.stack(torch.split(linear, self.n_labels))
        return out

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def _inner_step(self, batch, stage: str = None):

        """ Common actions for training, test and val steps. """

        # x[0] is the time series
        # x[1] are the sim frames
        
        x, y = batch
        x: torch.tensor
        y: torch.tensor

        output = self(x)
        loss = F.cross_entropy(output, y.to(torch.float32))

        # accumulate and return metrics for logging
        acc = self.__getattr__(f"{stage}_acc")(output, torch.argmax(y, dim=1))
        f1  = self.__getattr__(f"{stage}_f1")(output, torch.argmax(y, dim=1))
        
        if stage == "train":
            self.log(f"{stage}_loss", loss, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_f1", f1, prog_bar=True, sync_dist=True)

        return loss.to(torch.float32)

    def training_step(self, batch, batch_idx):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch, batch_idx):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        """ Test step. """
        return self._inner_step(batch, stage="test")

    # EPOCH END
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _custom_epoch_end(self, step_outputs: list[torch.Tensor], stage: str):

        """ Common actions for validation and test epoch ends. """

        # metrics to analyze
        metrics = ["acc", "f1"]

        # task flags
        if stage == "val":
            print("")
        print(f"\n\n  ~~ {stage} stats ~~")
        for metric in metrics:
            mstring = f"{stage}_{metric}"
            val = self.__getattr__(mstring).compute()
            if stage == "train":
                self.log("epoch_" + mstring, val, sync_dist=True)
            else:
                self.log(mstring, val, sync_dist=True)
            self.__getattr__(mstring).reset()
            print(f"{mstring}: {val:.4f}")
        print("")

    def training_epoch_end(self, training_step_outputs):
        """ Actions to carry out at the end of each training epoch. """
        self._custom_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        """ Actions to carry out at the end of each validation epoch. """
        self._custom_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        """ Actions to carry out at the end of each test epoch. """
        self._custom_epoch_end(test_step_outputs, "test")

    # OPTIMIZERS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def configure_optimizers(self):

        """ Define optimizers and LR schedulers. """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": "val_acc",
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }