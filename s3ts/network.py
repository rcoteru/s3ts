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
from s3ts.data_str import TaskParameters

import logging

log = logging.Logger(__name__)

# ========================================================= #
#                     MULTITASK MODEL                       #
# ========================================================= #

class MultitaskModel(LightningModule):

    def __init__(self,      
        n_labels: int,
        n_patterns: int, 
        patt_length: int,
        window_size: int,
        tasks: TaskParameters,
        max_feature_maps: int = 128,
        learning_rate: float = 1e-5
        ):

        super().__init__()

        self.n_labels = n_labels
        self.n_patterns = n_patterns
        self.patt_length = patt_length
        self.window_size = window_size
        
        self.tasks = tasks

        self.learning_rate = learning_rate

        # main encoder
        self.conv_encoder = ConvEncoder(
            in_channels=n_patterns,
            out_channels=max_feature_maps,
            conv_kernel_size=3,
            pool_kernel_size=3,
            img_height=patt_length,
            img_width=window_size
        )

        encoder_out_feats = self.conv_encoder.get_output_shape()

        # main classification
        self.main_decoder = nn.Sequential(
            LinSeq(in_features=encoder_out_feats,
            hid_features=encoder_out_feats*2,
            out_features=self.n_labels,
            hid_layers=0),
            nn.Softmax(dim=self.n_labels))

        # discretized classification
        self.disc_decoder = nn.Sequential(
            LinSeq(in_features=encoder_out_feats,
            hid_features=encoder_out_feats*2,
            out_features=tasks.discrete_intervals,
            hid_layers=0),
            nn.Softmax(dim=tasks.discrete_intervals))

        # discretized prediction decoder
        self.pred_decoder = nn.Sequential(
            LinSeq(in_features=encoder_out_feats,
            hid_features=encoder_out_feats*2,
            out_features=tasks.discrete_intervals,
            hid_layers=0),
            nn.Softmax(dim=tasks.discrete_intervals))
            
        # autoencoder
        self.conv_decoder = ConvDecoder(
            in_channels=max_feature_maps,
            out_channels=n_patterns,
            conv_kernel_size=3,
            img_height=patt_length,
            img_width=window_size,
            encoder_feats=encoder_out_feats)

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

        shared = self.conv_encoder(x)

        # auxiliary tasks
        main_out = self.main_decoder(shared)
        disc_out = self.disc_decoder(shared)
        pred_out = self.pred_decoder(shared)
        aenc_out = self.conv_decoder(shared)

        return main_out, disc_out, pred_out, aenc_out

    # STEPS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def predict_step(self, batch, batch_idx):

        """ Just calls forward, can be used to predict. """

        # this calls forward
        return self(batch)

    # TODO
    def _inner_step(self, x, y):
        
        main_out, disc_out, pred_out, aenc_out  = self(x)
        olabel, dlabel, dlabel_pred = y

        y_true_main = F.one_hot(olabel, num_classes=self.n_labels).type(torch.DoubleTensor)
        y_true_disc = F.one_hot(dlabel, num_classes=self.n_labels).type(torch.DoubleTensor)
        y_true_pred = F.one_hot(dlabel_pred, num_classes=self.n_labels).type(torch.DoubleTensor)

        main_loss = F.cross_entropy(main_out, y_true_main)
        disc_loss = F.cross_entropy(disc_out, y_true_disc)
        pred_loss = F.cross_entropy(pred_out, y_true_pred)
        aenc_loss = F.mse_loss(aenc_out, x)

        total_loss = np.average([main_loss, disc_loss, pred_loss, aenc_loss],
            weights=[5,1,1,1])

        return total_loss, main_out

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

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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