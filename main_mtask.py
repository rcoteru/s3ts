"""
Run the main classification task alongside in two scenarios: 
alone and with shifted discrete label pretrains.

@author Raúl Coterillo
"""

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

#from pytorch_lightning import 

from s3ts.data_str import AugProbabilities, TaskParameters
from s3ts.network import MultitaskModel
from s3ts.data import MTaskDataModule

import time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

EXPERIMENT = "prueba_CBF"
DATASET = "CBF"

TEST_SIZE = 0.3

STS_LENGTH = 40

WINDOW_SIZE = 10
BATCH_SIZE  = 128
LEARNING_RATE = 1E-5

RANDOM_STATE = 0
seed_everything(RANDOM_STATE)

probs = AugProbabilities()
tasks = TaskParameters(
    disc=True,
    pred=True,
)



print("Computing dataset...")
start_time = time.perf_counter()
dm = MTaskDataModule(
    experiment="test",
    dataset="GunPoint",
    sts_length=STS_LENGTH,
    window_size=WINDOW_SIZE,
    tasks=tasks,
    batch_size=BATCH_SIZE,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

print("Creating model...", end="")
start_time = time.perf_counter()
model = MultitaskModel(
    n_labels=dm.n_labels,
    n_patterns=dm.n_patterns,
    patt_length=dm.sample_length,
    window_size=WINDOW_SIZE,
    tasks=tasks,
    max_feature_maps=128,
    learning_rate=LEARNING_RATE)
end_time = time.perf_counter()
print(end_time - start_time, "seconds")

print("Setup the trainer...")
start_time = time.perf_counter()
lr_monitor = LearningRateMonitor(logging_interval='step')
model_checkpoint = ModelCheckpoint(dm.exp_path, save_last=True)
trainer = Trainer(default_root_dir=dm.exp_path,
    callbacks=[lr_monitor, model_checkpoint],
    max_epochs=100, check_val_every_n_epoch=1,
    deterministic = True)
end_time = time.perf_counter()
print(end_time - start_time, "seconds")

print("Begin training...")
trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm)
trainer.test(model, datamodule=dm)