"""

@author Raúl Coterillo
"""

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import torch

# nn training
from s3ts.network.architecture import CNN_DTW, model_wrapper
from s3ts.datasets.modules import ESM_DM

# metrics
from sklearn.metrics import classification_report

import numpy as np
import tqdm

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

# DATASET_NAME = "GunPoint"
DATASET_NAME = "CBF"

WINDOW_SIZE = 5

BATCH_SIZE = 128
LEARNING_RATE = 5E-3
MAX_EPOCHS = 15

RESTART_PATH = "/home/rcoterillo/proyectos/ODTWFrames/s3ts/data/CBF/checkpoints/epoch=14-step=4200-v1.ckpt"

NPROCS = 4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    # define data folder
    save_path = Path.cwd() / "data" / DATASET_NAME
    save_path.mkdir(parents=True, exist_ok=True)

    # create data modules
    dm_main = ESM_DM(save_path, window_size=WINDOW_SIZE, task="main-task",
                batch_size=BATCH_SIZE, num_workers=NPROCS)

    dm_aux = ESM_DM(save_path, window_size=WINDOW_SIZE, task="aux-task",
                batch_size=BATCH_SIZE, num_workers=NPROCS)

    # create the model
    model = model_wrapper(
        model_architecture=CNN_DTW,
        ref_size=dm_main.ds_train.ESMs.shape[2],
        channels=dm_main.channels,
        labels=dm_main.labels_size,
        window_size=WINDOW_SIZE,
        lr=LEARNING_RATE
    )

    chkpt_path = save_path / "checkpoints"

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(chkpt_path)

    # Initialize trainers
    trainer_main = Trainer(
        default_root_dir=save_path,
        callbacks=[lr_monitor, model_checkpoint],
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=1,
        #progress_bar_refresh_rate=30,
        deterministic=True,
    )

    trainer_aux = Trainer(
        default_root_dir=save_path,
        callbacks=[lr_monitor, model_checkpoint],
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=1,
        #progress_bar_refresh_rate=30,
        deterministic=True,
    )

    if RESTART_PATH is None:

        # Pretrain the model
        # trainer_aux.fit(model, datamodule=dm_aux)
        # trainer_aux.validate(model, datamodule=dm_aux)
        # trainer_aux.test(model, datamodule=dm_aux)

        # Train the model on main task
        trainer_main.fit(model, datamodule=dm_main)
        trainer_main.validate(model, datamodule=dm_main)
        trainer_main.test(model, datamodule=dm_main)

        path = chkpt_path / f"epoch={trainer_main.current_epoch-1}-step={trainer_main.global_step}.ckpt"

    else:
        path = RESTART_PATH

    model = model_wrapper.load_from_checkpoint(path,
                            model_architecture=CNN_DTW,
                            ref_size=dm_main.ds_test.ESMs.shape[2],
                            channels=dm_main.channels,
                            labels=dm_main.labels_size,
                            window_size=WINDOW_SIZE)

    model.eval()
    model.freeze()

    # evaluate network
    results_path = save_path / "results"
    total_len = len(dm_main.ds_test)

    y_pred = []
    y_true = []
    predict_dataloader = dm_main.test_dataloader()

    with torch.inference_mode():
        for i, (x, y) in tqdm.tqdm(enumerate(predict_dataloader), total=total_len // BATCH_SIZE):
            #x = x.cuda()
            raw_score = model(x)
            y_pred.extend(raw_score.softmax(dim=-1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    print('Classification Report')
    target_names = [str(i) for i in range(dm_main.labels_size)]
    print(classification_report(y_true, np.argmax(y_pred, axis=-1)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
