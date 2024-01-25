#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Wrapper model for the deep learning models. """

from __future__ import annotations

# modules
from pytorch_lightning import LightningModule

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
        in_dimensions: int,
        decoder_output_size: int,
        latent_dims: int,
        dropout: float = 0,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        feedforward_mult: int = 2
    ):
        super().__init__()

        self.embedding = nn.Linear( # takes input of shape (n, t, d) -> (n, t, decoder_output_size)
            in_features=in_dimensions,
            out_features=latent_dims
        )

        self.positional_encoder = PositionalEncoding(latent_dims, dropout=dropout, max_len=1000)

        self.decoder_output = nn.Linear(
            in_features=latent_dims,
            out_features=decoder_output_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dims,
            nhead=n_heads,
            dim_feedforward=latent_dims * feedforward_mult,
            batch_first=True,
        ) # (batch, seq, feature) -> (batch, seq, feature)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
        )

    def forward(
        self,
        src: Tensor,
        mask: Tensor=None
    ):
        src = self.embedding(src) # (n, t, d) -> (n, t, latent)
        src = self.positional_encoder(src) # (n, t, latent) -> (n, t, latent)

        src = self.encoder(
            src=src,
            mask=mask
        ) # (n, t, latent) -> (n, t, latent)

        # redue time dimension with global average pooling
        src = src.mean(dim=1) # (n, t, latent) -> (n, latent)

        return self.decoder_output(src) # (n, latent) -> (n, n_classes)

class TransformerWrapper(LightningModule):

    name: str           # model name

    wdw_len: int        # window length
    wdw_str: int        # window stride

    n_dims: int         # number of STS dimensions
    n_classes: int      # number of classes
    
    latent_dims: int      # encoder feature hyperparam
    transformer_layers: int
    feedforward_mult: int
    n_heads: int
    dropout: float
    lr: float           # learning rate

    def __init__(self, dsrc,
        n_dims, n_classes, wdw_len, wdw_str, latent_dims, transformer_layers, feedforward_mult, n_heads, dropout,
        lr, voting, weight_decayL1, weight_decayL2,
        name=None, args=None) -> None:

        """ Wrapper for the PyTorch models used in the experiments. """

        if name is None:
            name = f"transformer_{latent_dims}_{transformer_layers}_{feedforward_mult}_{n_heads}"

        # save parameters as attributes
        super().__init__(), self.__dict__.update(locals())
        self.save_hyperparameters()

        self.transformer = TimeSeriesTransformer(
            in_dimensions=n_dims, 
            decoder_output_size=n_classes, 
            latent_dims=latent_dims,
            dropout=dropout,
            n_heads=n_heads,
            n_encoder_layers=transformer_layers,
            feedforward_mult=feedforward_mult
        )

        # create softmax and flatten layers
        self.flatten = nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax()

        # create metrics
        for phase in ["train", "val", "test"]: 
            self.__setattr__(f"{phase}_cm", tm.ConfusionMatrix(num_classes=n_classes, task="multiclass"))
            if phase != "train":
                self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=n_classes, task="multiclass", average="macro"))

        self.voting = None
        if voting["n"] > 1:
            self.voting = voting
            self.voting["weights"] = (self.voting["rho"] ** (1/self.wdw_len)) ** torch.arange(self.voting["n"] - 1, -1, -1)

        self.previous_predictions = None
        self.probabilities = []
        self.labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """
        x=self.logits(x)
        x = self.softmax(x)
        return x
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # create mask
        mask = generate_square_subsequent_mask(x.shape[-1], x.shape[-1])

        # x input is dims (n, d, t), transformer input must be (n, t, d)
        x = x.transpose((0, 2, 1))
        x = self.transformer(x, mask=mask)
        return x

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        output = self.logits(batch["series"])

        # Compute the loss and metrics
        loss = F.cross_entropy(output, batch["label"])

        if stage == "train" or self.voting is None:
            predictions = torch.argmax(output, dim=1)
        if stage != "train" and not self.voting is None:
            pred_prob = torch.softmax(output, dim=1)

            if self.previous_predictions is None:
                pred_ = torch.cat((torch.zeros((self.voting["n"]-1, self.n_classes)), pred_prob), dim=0)
            else:
                pred_ = torch.cat((self.previous_predictions, pred_prob), dim=0)
            
            self.previous_predictions = pred_prob[-(self.voting["n"]-1):,:]

            predictions_weighted = torch.conv2d(pred_[None, None, ...], self.voting["weights"][None, None, :, None])[0, 0]
            predictions = predictions_weighted.argmax(dim=1)

            self.probabilities.append(pred_prob)
            self.labels.append(batch["label"])

        self.__getattr__(f"{stage}_cm").update(predictions, batch["label"])
        if stage != "train" and self.voting is None:
            self.probabilities.append(torch.softmax(output, dim=1))
            self.labels.append(batch["label"])

            # auroc = self.__getattr__(f"{stage}_auroc")(output, batch["label"])  

        # log loss and metrics
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        # return loss
        if stage == "train":
            l1_loss = torch.tensor(0., requires_grad=True)
            l2_loss = torch.tensor(0., requires_grad=True)
            if self.weight_decayL1 > 0:
                l1_loss = self.weight_decayL1 * sum(p.abs().sum() for name, p in self.named_parameters() if ("bias" not in name and "bn" not in name))
                # self.log(f"{stage}_L1", l1_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

            if self.weight_decayL2 > 0:
                l2_loss = self.weight_decayL2 * sum(p.square().sum() for name, p in self.named_parameters() if ("bias" not in name and "bn" not in name))
                # self.log(f"{stage}_L2", l2_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

            return loss.to(torch.float32) + l1_loss.to(torch.float32) + l2_loss.to(torch.float32)

        return loss.to(torch.float32)

    def training_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Test step. """
        return self._inner_step(batch, stage="test")
    
    def log_metrics(self, stage):
        cm = self.__getattr__(f"{stage}_cm").compute()
        self.__getattr__(f"{stage}_cm").reset()

        TP = cm.diag()
        FP = cm.sum(0) - TP
        FN = cm.sum(1) - TP
        TN = torch.empty(cm.shape[0])
        for i in range(cm.shape[0]):
            TN[i] = cm[:i,:i].sum() + cm[:i,i:].sum() + cm[i:,:i].sum() + cm[i:,i:].sum()

        precision = TP/(TP+FP)
        recall = TP/(TP+FN) # this is the same as accuracy per class
        f1 = 2*(precision*recall)/(precision + recall)
        iou = TP/(TP+FP+FN) # iou per class

        self.log(f"{stage}_pr", precision.nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"{stage}_re", recall.nanmean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"{stage}_f1", f1.nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"{stage}_iou", iou.nanmean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)

        if stage != "train":
            auc_per_class = tm.functional.auroc(
                torch.concatenate(self.probabilities, dim=0), 
                torch.concatenate(self.labels, dim=0), 
                task="multiclass",
                num_classes=self.n_classes)
            self.probabilities = []
            self.labels = []

            self.log(f"{stage}_auroc", auc_per_class.nanmean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)

    def on_train_epoch_end(self):
        self.log_metrics("train")

    def on_validation_epoch_end(self):
        self.log_metrics("val")

    def on_test_epoch_end(self):
        self.log_metrics("test")

    def predict_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Predict step. """
        output = self(batch["series"])
        return output

    def configure_optimizers(self):
        """ Configure the optimizers. """
        mode = "max"
        monitor = "val_re"
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 
                    mode=mode, factor=np.sqrt(0.1), patience=2, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": monitor,
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }