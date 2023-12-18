#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Wrapper model for the deep learning models. """

from __future__ import annotations

# modules
from pytorch_lightning import LightningModule
from s3ts.api.nets.encoders.frames.CNN import CNN_IMG
from s3ts.api.nets.encoders.frames.RES import RES_IMG
from s3ts.api.nets.encoders.series.RNN import RNN_TS
from s3ts.api.nets.encoders.series.CNN import CNN_TS
from s3ts.api.nets.encoders.series.RES import RES_TS
from s3ts.api.nets.encoders.frames.simpleCNN import SimpleCNN_IMG
from s3ts.api.nets.encoders.series.simpleCNN import SimpleCNN_TS
from s3ts.api.nets.decoders.linear import LinearDecoder
from s3ts.api.nets.decoders.mlp import MultiLayerPerceptron

from s3ts.api.nets.encoders.dtw.dtw_layer import DTWLayer

# base torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchmetrics as tm
import torch.nn as nn
import numpy as np
import torch

encoder_dict = {"img": {"cnn": CNN_IMG, "res": RES_IMG, "simplecnn": SimpleCNN_IMG},
    "ts": {"rnn": RNN_TS, "cnn": CNN_TS, "res": RES_TS, "simplecnn": SimpleCNN_TS},
    "dtw": {"cnn": CNN_IMG, "res": RES_IMG, "simplecnn": SimpleCNN_IMG}}

decoder_dict = {"linear": LinearDecoder, "mlp": MultiLayerPerceptron}

class WrapperModel(LightningModule):

    name: str           # model name
    dsrc: str          # input dsrc: ["img", "ts"]
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

    def __init__(self, dsrc, arch, dec_arch, task,
        n_dims, n_classes, n_patterns, l_patterns,
        wdw_len, wdw_str, sts_str,
        enc_feats, dec_feats, dec_layers, lr, voting,
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
        elif dsrc == "dtw":
            self.dtw_layer = DTWLayer(n_patts=enc_feats, d_patts=self.n_dims, l_patts=l_patterns, l_out=l_patterns, rho=0.01)
            ref_size, channels = l_patterns, enc_feats
            self.wdw_len = ref_size

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
        self.decoder = decoder_dict[dec_arch](inp_feats=inp_feats, 
            hid_feats=dec_feats, out_feats=out_feats, hid_layers=dec_layers)

        # create softmax and flatten layers
        self.flatten = nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax()

        # create metrics
        if self.task == "cls":
            for phase in ["train", "val", "test"]: 
                self.__setattr__(f"{phase}_cm", tm.ConfusionMatrix(num_classes=out_feats, task="multiclass"))
                if phase != "train":
                    self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=out_feats, task="multiclass", average="macro"))
        elif self.task == "reg":
            for phase in ["train", "val", "test"]:
                self.__setattr__(f"{phase}_mse", tm.MeanSquaredError(squared=False))
                self.__setattr__(f"{phase}_r2",  tm.R2Score(num_outputs=out_feats))

        self.voting = None
        if voting["n"] > 1:
            self.voting = voting
            self.voting["weights"] = self.voting["w"] ** torch.arange(self.voting["n"], 0, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """
        if self.dsrc == "dtw":
            x = self.dtw_layer(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)
        if self.task == "cls":
            x = self.softmax(x)
        if self.task == "reg":
            bsize = x.shape[0]
            x = x.reshape((bsize, self.n_dims, -1))
        return x
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.dsrc == "dtw":
            x = self.dtw_layer(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)
        return x

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """

        # Forward pass
        if self.dsrc == "img":
            output = self.logits(batch["frame"])
        elif self.dsrc == "ts" or self.dsrc == "dtw":
            output = self.logits(batch["series"])

        # Compute the loss and metrics
        if self.task == "cls":
            loss = F.cross_entropy(output, batch["label"])

            predictions = torch.argmax(output, dim=1)
            if not self.voting is None:
                predictions_one_hot = torch.eye(self.n_classes)[predictions] # equivalent to F.one_hot(predictions, self.n_classes).float()
                predictions_weighted = torch.conv2d(
                    F.pad(predictions_one_hot[None, None, ...], (0, 0, self.voting["n"]-1, 0)),
                    self.voting["weights"][None, None, :, None]
                )[0, 0]
                predictions = torch.argmax(predictions_weighted, dim=1)

            self.__getattr__(f"{stage}_cm").update(predictions, batch["label"])
            if stage != "train":
                auroc = self.__getattr__(f"{stage}_auroc")(output, batch["label"])  
        elif self.task == "reg":
            loss = F.mse_loss(output, batch["series"])
            mse = self.__getattr__(f"{stage}_mse")(output, batch["series"])
            r2 = self.__getattr__(f"{stage}_r2")(self.flatten(output), 
                                                 self.flatten(batch["series"]))

        # log loss and metrics
        on_step = True if stage == "train" else False
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)
        if self.task == "cls":
            #self.log(f"{stage}_acc", self.__getattr__(f"{stage}_acc"), on_epoch=True, on_step=False, prog_bar=True, logger=True)
            #self.log(f"{stage}_f1", self.__getattr__(f"{stage}_f1"), on_epoch=True, on_step=False, prog_bar=False, logger=True)
            if stage != "train":
                self.log(f"{stage}_auroc", self.__getattr__(f"{stage}_auroc"), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        elif self.task == "reg":
            self.log(f"{stage}_mse", mse, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            self.log(f"{stage}_r2", r2, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        # return loss
        return loss.to(torch.float32)

    def training_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def test_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Validation step. """
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

        self.log(f"{stage}_pr", precision.mean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"{stage}_re", recall.mean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"{stage}_f1", f1.mean(), on_epoch=True, on_step=False, prog_bar=False, logger=True)

    def on_train_epoch_end(self):
        self.log_metrics("train")

    def on_validation_epoch_end(self):
        self.log_metrics("val")

    def on_test_epoch_end(self):
        self.log_metrics("test")

    def predict_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Predict step. """
        if self.dsrc == "img":
            output = self(batch["frame"])
        elif self.dsrc == "ts":
            output = self(batch["series"])
        return output

    def configure_optimizers(self):
        """ Configure the optimizers. """
        mode = "max" if self.task == "cls" else "min"
        monitor = "val_re" if self.task == "cls" else "val_mse"
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