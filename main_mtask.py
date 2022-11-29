"""
Run the main classification task alongside in two scenarios: 
alone and with shifted discrete label pretrains.

@author Raúl Coterillo
"""

from s3ts.data import MTaskDataModule, AuxTasksParams
from s3ts.data_aux import AugProbabilities

from s3ts.network import MultitaskModel

import time

# User Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

EXPERIMENT = "prueba_CBF"
DATASET = "CBF"

TEST_SIZE = 0.3

WINDOW_SIZE = 5
BATCH_SIZE  = 128
LEARNING_RATE = 1E-5

MAIN_STS_LENGTH = 20
AUX_STS_LENGTH  = 200

probs = AugProbabilities()
tasks = AuxTasksParams()

print("Computing dataset...")
start_time = time.perf_counter()
dm = MTaskDataModule(
    experiment="test",
    dataset="GunPoint",
    sts_length=40,
    window_size=WINDOW_SIZE,
    tasks=tasks,
    batch_size=BATCH_SIZE,
    test_size=0.3,
    random_state=0)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")


print("Creating model...", end="")
start_time = time.perf_counter()


model = MultitaskModel(
    n_labels=dm.n_labels,
    n_patterns=dm.n_patterns,
    n_patterns=
    )


end_time = time.perf_counter()
print(end_time - start_time, "seconds")
