"""
Multitask Learning Model

@version 2022-12
@author Raúl Coterillo
"""

# lightning
from pytorch_lightning import LightningModule

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torch.nn as nn
import torchmetrics
import torch

# numpy
import numpy as np

from s3ts.network_aux import ConvEncoder, ConvDecoder, LinSeq

import logging

log = logging.Logger(__name__)

# ========================================================= #
#                     MULTITASK MODEL                       #
# ========================================================= #

class MultitaskModel(LightningModule):

    def __init__(self, 
        encoder: ConvEncoder,    
        patt_size: int, 
        n_patterns: int, 
        ds_labels: int,
        tasks: list[str, tuple],
        main_task_only: bool = True):

        super().__init__()

        self.patt_size = patt_size
        self.n_patterns = n_patterns
        self.main_task_only = main_task_only
        self.tasks = tasks

        self.encoder = encoder
        self.decoder = nn.Linear(in_features=self.encoder.get_output_shape(), 
                out_features=ds_labels)

        self.aux_decoders = []
        for t in tasks:
            if t[1] == "cls":
                meta = t[2]
                tasks.append(nn.Linear(in_features=self.encoder.get_output_shape(), 
                    out_features=meta[1]))
            if t[1] == "ae":
                tasks.append(ConvDecoder(
                    in_channels=encoder.out_channels,
                    out_channels=encoder.in_channels,
                    conv_kernel_size=encoder.conv_kernel_size,
                    img_height=encoder.img_height,
                    img_width=encoder.img_width,
                    encoder_feats=encoder.get_encoder_features()))
            else:
                raise NotImplementedError

        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(num_classes=n_patterns, average="micro")
        self.train_auroc = torchmetrics.AUROC(num_classes=n_patterns, average="macro")

        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1Score(num_classes=n_patterns, average="micro")
        self.val_auroc = torchmetrics.AUROC(num_classes=n_patterns, average="macro")

        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1Score(num_classes=n_patterns, average="micro")
        self.test_auroc = torchmetrics.AUROC(num_classes=n_patterns, average="macro")

    # TODO
    def get_output_shape(self):
        return None

    def forward(self, x):

        """ Use for inference only (separate from training_step)"""

        shared = self.encoder(x)
        output = self.decoder(shared) 

        if self.main_task_only:
            return output

        aux_outputs = []
        for i in range(len(self.tasks)):
            aux_outputs.append(self.decoders[i](shared))

        return output, aux_outputs

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # TODO
    def _inner_step(self, x, y):
        
        logits, decoded = self(x)
        
        y_pred = logits.softmax(dim=-1)

        y_true = F.one_hot(y, num_classes=self.labels)
        y_true = y_true.type(torch.DoubleTensor)
        
        main_loss = F.cross_entropy(logits, )
        
        return loss, y_pred

    # TODO
    def training_step(self, batch, batch_idx):

        """ Complete training loop. """

        self.main_task_only = False

        x, y = batch
        loss, y_pred = self._inner_step(x, y)

        # accumulate and return metrics for logging
        acc = self.train_acc(y_pred, y)
        f1 = self.train_f1(y_pred, y)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, sync_dist=True)
        self.log("train_f1", f1, prog_bar=True, sync_dist=True)

        return loss
    
    def _custom_stats_step(self, batch, stage=None):
        
        """ Common actions for validation and test step. """

        x, y = batch
        loss, y_pred = self._inner_step(x, y)

        if stage == 'val':
            self.val_acc(y_pred, y)
            self.val_f1(y_pred, y)
            self.val_auroc(y_pred, y)

        elif stage == "test":
            self.test_acc(y_pred, y)
            self.test_f1(y_pred, y)
            self.test_auroc(y_pred, y)

        return loss
    
    def validation_step(self, batch, batch_idx):
        return self._custom_stats_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._custom_stats_step(batch, "test")

    # EPOCH END
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # TODO
    def training_epoch_end(self, training_step_outputs):

        """ Actions to carry out at the end of each epoch. """
        
        # compute metrics
        train_accuracy = self.train_acc.compute()
        train_f1 = self.train_f1.compute()

        # log metrics
        self.log("epoch_train_accuracy", train_accuracy, prog_bar=True, sync_dist=True)
        self.log("epoch_train_f1", train_f1, prog_bar=True, sync_dist=True)

        # reset all metrics
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()
        print(f"\ntraining accuracy: {train_accuracy:.4}, " f"f1: {train_f1:.4}")

    def _custom_epoch_end(self, step_outputs, stage):

        """ Common actions for validation and test epoch ends. """

        if stage == "val":
            acc_metric = self.val_acc
            f1_metric = self.val_f1
            auroc_metric = self.val_auroc
        elif stage == "test":
            acc_metric = self.test_acc
            f1_metric = self.test_f1
            auroc_metric = self.test_auroc

        # compute metrics
        loss = torch.tensor(step_outputs).mean()
        accuracy = acc_metric.compute()
        f1 = f1_metric.compute()
        auroc = auroc_metric.compute()

        # log metrics
        self.log(f"{stage}_accuracy", accuracy, sync_dist=True)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        self.log(f"{stage}_f1", f1, sync_dist=True)
        self.log(f"{stage}_auroc", auroc, sync_dist=True)

        # reset all metrics
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        print(f"\n{stage} accuracy: {accuracy:.4} " f"f1: {f1:.4}, auroc: {auroc:.4}")

    def validation_epoch_end(self, validation_step_outputs):
        self._custom_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        self._custom_epoch_end(test_step_outputs, "test")

    # OPTIMIZERS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # TOCHECK
    def configure_optimizers(self):

        """ Define optimizers and LR schedulers. """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }