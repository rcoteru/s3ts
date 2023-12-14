#/usr/bin/python3
# -*- coding: utf-8 -*-

""" Common functions for the experiments. """

# models / modules
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
import torch

# in-package imports
from s3ts.models.neighbors import knn_dtw_predict
from s3ts.models.wrapper import WrapperModel
from s3ts.legacy.modules import DFDataModule

# standard library
from pathlib import Path
import logging as log

# basics
import pandas as pd

def setup_trainer(
        version: str,
        directory: Path,
        max_epochs: int, 
        stop_metric: str, 
        stop_mode: str, 
        ) -> tuple[Trainer, ModelCheckpoint]:
    
    """ Setup the trainer. """
    
    # Create the callbacks
    checkpoint = ModelCheckpoint(monitor=stop_metric, mode=stop_mode)    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    #early_stop = EarlyStopping(monitor=stop_metric, mode=stop_mode, patience=20)
    callbacks = [lr_monitor, checkpoint]#, early_stop]
    # Create the loggers
    tb_logger = TensorBoardLogger(save_dir=directory, name="logs", version=version)
    csv_logger = CSVLogger(save_dir=directory, name="logs", version=version)
    loggers = [tb_logger, csv_logger]
    # Create the trainer
    return Trainer(default_root_dir=directory,  accelerator="auto",
    logger=loggers, callbacks=callbacks,
    max_epochs=max_epochs,  benchmark=True, deterministic=False, 
    log_every_n_steps=1, check_val_every_n_epoch=1), checkpoint

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_model(pretrain_mode: bool, version: str,
        dataset: str, mode: str, arch: str, dm: DFDataModule, 
        directory: Path, max_epochs: int, learning_rate: float,
        num_encoder_feats: int, num_decoder_feats: int,
        encoder_path: Path, cv_rep: int, random_state: int,
        ) -> tuple[dict, WrapperModel]:
    
    # Set the random seed
    seed_everything(random_state, workers=True)

    # Save common parameters
    res = dict()
    
    res["mode"] = mode
    res["arch"] = arch
    res["dataset"] = dataset

    if arch == "nn":

        acc, f1, model = knn_dtw_predict(dm=dm, metric="dtw", n_neighbors=1)
        
        # this would be the event size
        res["window_length"] = dm.window_length

        res["pretrained"] = False
        res["test_acc"] = acc
        res["test_f1"] = f1
        res["cv_rep"] = cv_rep

    else:

        res["val_size"] = dm.val_size
        res["max_epochs"] = max_epochs
        res["batch_size"] = dm.batch_size
        res["pretrain_mode"] = pretrain_mode
        res["window_length"] = dm.window_length
        res["stride_series"] = dm.stride_series
        res["window_time_stride"] = dm.window_time_stride
        res["window_patt_stride"] = dm.window_patt_stride
        res["learning_rate"] = learning_rate
        res["random_state"] = random_state

        # Create a label for the experiment

        if pretrain_mode:

            metrics = ["mse", "r2"]
            trainer, ckpt = setup_trainer(directory=directory, version=version,
            max_epochs=max_epochs, stop_metric="val_mse", stop_mode="min")
            
            model = WrapperModel(mode="df", arch=arch, target="reg",
                n_classes=dm.n_classes, window_length=dm.window_length, 
                n_patterns=dm.n_patterns, l_patterns=dm.l_patterns,
                window_time_stride=dm.window_time_stride, window_patt_stride=dm.window_patt_stride,
                stride_series=dm.stride_series, learning_rate=learning_rate,
                encoder_feats=num_encoder_feats, decoder_feats=num_decoder_feats)
            
        else:

            metrics = ["acc", "f1", "auroc"]
            trainer, ckpt = setup_trainer(directory=directory, version=version,
            max_epochs=max_epochs, stop_metric="val_acc", stop_mode="max")
            model = WrapperModel(mode=mode, arch=arch, target="cls",
                n_classes=dm.n_classes, window_length=dm.window_length, 
                n_patterns=dm.n_patterns, l_patterns=dm.l_patterns,
                window_time_stride=dm.window_time_stride, window_patt_stride=dm.window_patt_stride,
                stride_series=dm.stride_series, learning_rate=learning_rate,
                encoder_feats=num_encoder_feats, decoder_feats=num_decoder_feats)
            
            # Load the encoder if needed
            if encoder_path is not None:
                model.encoder = torch.load(encoder_path)
                res["pretrained"] = True
            else:
                res["pretrained"] = False
                res.pop("stride_series")
            res["cv_rep"] = cv_rep

        # TODO: uncomment when its not so buggy, supposed to improve performance
        # model: torch.Module = torch.compile(model, mode="reduce-overhead")

        trainer.fit(model=model, datamodule=dm)
        model = model.load_from_checkpoint(ckpt.best_model_path)

        # Save training results
        res["nepochs"] = trainer.current_epoch
        res["best_model"] = ckpt.best_model_path
        res["total_params"] = sum(p.numel() for p in model.parameters())
        res["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        res["metrics_csv"] = str(directory  / "logs" / version / "metrics.csv")
        data = trainer.validate(model, datamodule=dm)
        for m in metrics:
            res[f"val_{m}"]  = data[0][f"val_{m}"]        
        if not pretrain_mode:
            data = trainer.test(model, datamodule=dm)
            for m in metrics:
                res[f"test_{m}"] = data[0][f"test_{m}"]

        if pretrain_mode:
            # Save the pretrained encoder
            log.info(f"Saving encoder at: '{str(encoder_path)}'")
            torch.save(model.encoder, encoder_path)

    return res, model

def save_results(res: dict, res_fname: str, storage_dir: Path):

    results_path = storage_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    res_file = results_path / res_fname

    # Create a dataframe with the results, using the keys as columns
    res_df = pd.DataFrame(res, index=[0])

    # Read the results file and append the new results to it
    if res_file.exists():
        old_res_df = pd.read_csv(res_file)
        res_df = pd.concat([old_res_df, res_df], ignore_index=True)

    # Save the results 
    res_df.to_csv(res_file, index=False)

    # Return the whole dataframe
    return res_df
