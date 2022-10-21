"""

@author Raúl Coterillo
"""

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import torch

# nn training
from s3ts.network.architecture import CNN_DTW, model_wrapper
from s3ts.datasets.modules import ESM_DM

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# plotting
import matplotlib.pyplot as plt
import tqdm

import numpy as np

from pathlib import Path
import warnings
import logging
import sys

seed_everything(0, workers=True)
warnings.filterwarnings("ignore", category=UserWarning) 
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s')

# Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

DATASET_NAME = "GunPoint"
# DATASET_NAME = "CBF"

WINDOW_SIZE = 5

BATCH_SIZE = 128
LEARNING_RATE = 5E-3
MAX_EPOCHS = 15

RESTART_PATH = "/home/rcoterillo/proyectos/ODTWFrames/s3ts/data/GunPoint/checkpoints/epoch=14-step=4920.ckpt"

NPROCS = 4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    # define data folder
    save_path = Path.cwd() / "data" / DATASET_NAME
    save_path.mkdir(parents=True, exist_ok=True)

    # create data module
    dm = ESM_DM(save_path, window_size=WINDOW_SIZE, task="main-task",
                batch_size=BATCH_SIZE, num_workers=NPROCS)
    dm.prepare_data()

    # create the model
    model = model_wrapper(
        model_architecture=CNN_DTW,
        ref_size=dm.ds_train.ESMs.shape[2], # danger, esto era un 1
        channels=dm.channels,
        labels=dm.labels_size,
        window_size=WINDOW_SIZE,
        lr=LEARNING_RATE
    )

    chkpt_path = save_path / "checkpoints"

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(chkpt_path)

    # Initialize a trainer
    trainer = Trainer(
        default_root_dir=save_path,
        callbacks=[lr_monitor, model_checkpoint],
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=1,
        #progress_bar_refresh_rate=30,
        deterministic=True,
    )

    if RESTART_PATH is None:
        # Train the model ⚡
        trainer.fit(model, datamodule=dm)
        trainer.validate(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
        path = chkpt_path / f"epoch={trainer.current_epoch -1}-step={trainer.global_step}.ckpt"
    else:
        path = RESTART_PATH

    model = model_wrapper.load_from_checkpoint(path,
                            model_architecture=CNN_DTW,
                            ref_size=dm.ds_test.ESMs.shape[2],
                            channels=dm.channels,
                            labels=dm.labels_size,
                            window_size=WINDOW_SIZE)

    model.eval()
    model.freeze()

    # evaluate network
    results_path = save_path / "results"
    total_len = len(dm.ds_test)

    y_pred = []
    y_true = []
    predict_dataloader = dm.test_dataloader()

    with torch.inference_mode():
        for i, (x, y) in tqdm.tqdm(enumerate(predict_dataloader), total=total_len // BATCH_SIZE):
            #x = x.cuda()
            raw_score = model(x)
            y_pred.extend(raw_score.softmax(dim=-1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    print('Classification Report')
    target_names = [str(i) for i in range(dm.labels_size)]
    print(classification_report(y_true, np.argmax(y_pred, axis=-1)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
