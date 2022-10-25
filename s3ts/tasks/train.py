"""
Automation of the training tasks.
"""

from pytorch_lightning.callbacks import LearningRateMonitor 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, LightningModule

from s3ts.datasets.modules import ESM_DM

from pathlib import Path 
import logging

log = logging.Logger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def run_sequence(
    exp_path: Path,
    seq_name: str,
    main_task: tuple[str, ESM_DM],
    aux_tasks: list[tuple[str, ESM_DM]],
    model: LightningModule
    ) -> LightningModule:

    seq_folder = exp_path / seq_name
    seq_folder.mkdir(exist_ok=True, parents=True)

    # if main task is finished, just load it
    target_file = seq_folder / main_task[0] / "last.ckpt"
    if target_file.exists():
        return target_file

    task_list = aux_tasks + [main_task]
    for task_idx, (task, task_dm) in enumerate(task_list):

        task_folder = seq_folder / task
        task_folder.mkdir(exist_ok=True, parents=True)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        model_checkpoint = ModelCheckpoint(task_folder, save_last=True)

        trainer = Trainer(
            default_root_dir=task_folder,
            callbacks=[lr_monitor, model_checkpoint],
            max_epochs=10, check_val_every_n_epoch=1,
            deterministic = True)

        trainer.fit(model, datamodule=task_dm)
        trainer.validate(model, datamodule=task_dm)
        trainer.test(model, datamodule=task_dm)

        # if auxiliary task, move on
        if task_idx + 1 != len(task_list):
            continue

        pass

    target_file = seq_folder / main_task[0] / "last.ckpt"
    return target_file

def print_task_info(
        exp_path: Path, 
        seq_name: str, 
        main_task: tuple[str, ESM_DM]
        ) -> None:


    pass

    