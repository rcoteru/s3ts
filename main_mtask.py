"""
Run the main classification task alongside in two scenarios: 
alone and with shifted discrete label pretrains.

@author Raúl Coterillo
"""

from s3ts.data import MTaskDataModule, AuxTasksParams
from s3ts.data_aux import AugProbabilites

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

aug_probs = AugProbabilites()
aux_tasks = AuxTasksParams()

# create data module
dm = MTaskDataModule()
