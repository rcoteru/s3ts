
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger



from s3ts.models.base import BasicModel


def pretrain_encoder(
    n_labels: int,
    n_patterns: int, 
    l_patterns: int,
    window_size: int,
    arch: type[LightningModule],
    dm: LightningDataModule,
    ) -> LightningModule:


    # 1. create model used for pretrain

    model = BasicModel(dm.n_labels)

    # 3. train the model

    # logger 
    tb_logger = TensorBoardLogger(save_dir="", name="lightning_logs")

    # early stop the model
    early_stop = EarlyStopping(monitor="val_auroc", mode="max", patience=5)

    # logger for learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # save checkpoints for the model
    model_checkpoint = ModelCheckpoint(pretrain_dm.exp_path, save_last=True)
    
    trainer = Trainer(default_root_dir=pretrain_dm.exp_path,
        logger = tb_logger, 
        callbacks=[lr_monitor, model_checkpoint, early_stop],
        max_epochs=100, 
        log_every_n_steps=1, check_val_every_n_epoch=1,
        deterministic = True)

    # 3. train the model

    trainer.fit(pretrain_model, datamodule=pretrain_dm)

    # 3. return the pretrained encoder

    encoder: LightningModule 

    return encoder