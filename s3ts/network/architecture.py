from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

from torch.nn import functional as F
import torch.nn as nn
import torchmetrics
import torch

import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class CNN_DTW(LightningModule):

    def __init__(self, ref_size, channels, window_size):
        super().__init__()

        self.channels = channels

        # main parameter to tune network complexity
        self.n_feature_maps = 32
        self.kernel_size = 3

        # convolutional part of the network
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.n_feature_maps // 2, 
                kernel_size=self.kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=self.n_feature_maps // 2, out_channels=self.n_feature_maps, 
                kernel_size=self.kernel_size, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Dropout(0.35),

            nn.Conv2d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2, 
                kernel_size=self.kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=self.n_feature_maps * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Conv2d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 4, 
                kernel_size=self.kernel_size, padding='same'),
        )

        # calculate cnn output shape
        image_dim = (1, channels, ref_size, window_size)
        features = self.model(torch.rand(image_dim).float())
        nfeats_after_conv = features.view(features.size(0), -1).size(1)

        # linear part 
        self.linear_1 = nn.Linear(in_features=nfeats_after_conv, out_features=self.n_feature_maps * 4)
        self.linear_2 = nn.Linear(in_features=self.n_feature_maps * 4, out_features=self.n_feature_maps * 8)

    def get_output_shape(self):
        return self.n_feature_maps * 8

    def forward(self, x):
        features = self.model(x.float())
        flat = features.view(features.size(0), -1)
        lin_1 = self.linear_1(flat)
        return self.linear_2(lin_1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class model_wrapper(LightningModule):

    def __init__(self, model_architecture, ref_size, channels, labels, window_size, lr=0.0001):
        
        """ Define computations here. """
        
        super().__init__()

        self.channels = channels
        self.labels = labels
        self.lr = lr

        self.model = model_architecture(ref_size, channels, window_size)
        
        self.classifier = nn.Linear(in_features=self.model.get_output_shape(), out_features=labels)

        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(num_classes=labels, average="micro")
        self.train_auroc = torchmetrics.AUROC(num_classes=labels, average="macro")

        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1Score(num_classes=labels, average="micro")
        self.val_auroc = torchmetrics.AUROC(num_classes=labels, average="macro")

        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1Score(num_classes=labels, average="micro")
        self.test_auroc = torchmetrics.AUROC(num_classes=labels, average="macro")

    def forward(self, x):

        """ Use for inference only (separate from training_step)"""

        feature = self.model(x.float())
        flat = feature.view(feature.size(0), -1)
        return self.classifier(flat)

    def _inner_step(self, x, y):
        logits = self(x)
        y_pred = logits.softmax(dim=-1)
        target = F.one_hot(y, num_classes=self.labels)
        target = target.type(torch.DoubleTensor)
        loss = F.cross_entropy(logits, target)
        return loss, y_pred

    def training_step(self, batch, batch_idx):

        """ Complete training loop. """

        x, y = batch
        loss, y_pred = self._inner_step(x, y)

        # accumulate and return metrics for logging
        acc = self.train_acc(y_pred, y)
        f1 = self.train_f1(y_pred, y)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=True, sync_dist=True)
        self.log("train_f1", f1, prog_bar=True, sync_dist=True)

        return loss

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
