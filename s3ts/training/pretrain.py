
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def pretrain_encoder(
    arch: type[LightningModule],
    dm: LightningDataModule,
    ) -> LightningModule:


    # 1. create model used for pretrain

    model = 

    # 3. train the model

    early_stop = EarlyStopping(monitor="val_auroc", mode="max", patience=5)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(pretrain_dm.exp_path, save_last=True)
    
    trainer = Trainer(default_root_dir=pretrain_dm.exp_path,
        callbacks=[lr_monitor, model_checkpoint, early_stop],
        max_epochs=100, check_val_every_n_epoch=1,
        deterministic = True)

    # 3. train the model

    trainer.fit(pretrain_model, datamodule=pretrain_dm)

    # 3. return the pretrained encoder

    encoder: LightningModule 

    return encoder