"""
Automation of the training tasks.
"""

from pytorch_lightning import Trainer, LightningModule

from s3ts.datasets.modules import ESM_DM

import logging

log = logging.Logger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def run_sequence(
        exp_path: Path,
        seq_name: str,
        main_task: tuple[str, ESM_DM],
        aux_tasks: list[tuple[str, ESM_DM]],
        model: LightningModule
        ) -> None:

    seq_folder = exp_path / seq_name

    for task, task_dm in aux_tasks:
        pass

    trainer_main = Trainer(
        default_root_dir=save_path,
        callbacks=[lr_monitor, model_checkpoint],
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=1,
        #progress_bar_refresh_rate=30,
        deterministic=True,
    )


    pass