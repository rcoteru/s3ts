import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics as tm

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

class lmodel(LightningModule):

    def __init__(self, model: torch.nn.Module, num_classes: int, lr: float = 0.001) -> None:

        name = f"Test model for DTW-layer"

        self.n_classes = num_classes
        # save parameters as attributes
        super().__init__()

        # select model architecture class
        self.model = model

        self.softmax = torch.nn.Softmax()

        self.lr = lr
        # create metrics
        for phase in ["train", "val", "test"]: 
            self.__setattr__(f"{phase}_acc", tm.Accuracy(num_classes=num_classes, task="multiclass"))
            self.__setattr__(f"{phase}_f1",  tm.F1Score(num_classes=num_classes, task="multiclass"))
            if phase != "train":
                self.__setattr__(f"{phase}_auroc", tm.AUROC(num_classes=num_classes, task="multiclass"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """
        x = self.model(x)
        x = self.softmax(x)

        return x

    def _inner_step(self, batch: dict[str: torch.Tensor], stage: str = None):

        """ Inner step for the training, validation and testing. """
        output = self.model(batch["series"])
        loss = F.cross_entropy(output, batch["label"])

        # compute metrics
        acc = self.__getattr__(f"{stage}_acc")(output, batch["label"])
        f1  = self.__getattr__(f"{stage}_f1")(output, batch["label"])
        if stage != "train":
            auroc = self.__getattr__(f"{stage}_auroc")(output, batch["label"])  

        # log loss and metrics
        on_step = True if stage == "train" else False
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        self.log(f"{stage}_acc", acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(f"{stage}_f1", f1, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        if stage != "train":
            self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        # return loss
        return loss

    def training_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Training step. """
        return self._inner_step(batch, stage="train")
        
    def validation_step(self, batch: dict[str: torch.Tensor], batch_idx: int):
        """ Validation step. """
        return self._inner_step(batch, stage="val")

    def configure_optimizers(self):
        """ Configure the optimizers. """
        mode = "max"
        monitor = "val_acc"
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