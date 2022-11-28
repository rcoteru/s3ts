"""
Run the main classification task alongside in two scenarios: 
alone and with shifted discrete label pretrains.

@author Raúl Coterillo
"""

from s3ts import plots
from s3ts.network.architecture import model_wrapper, CNN_DTW
from s3ts.tasks import data, train
import s3ts

from copy import deepcopy
import logging

s3ts.RANDOM_STATE = 0
log = logging.Logger(__name__)
logging.basicConfig(level=logging.INFO)

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

SHIFTS = [0]

AUX_TASKS = [
    # name, type, settings
    ("ae", "ae", ()),
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    # create experiment folder
    exp_path = data.prepare_experiment(EXPERIMENT)
    
    # download dataset
    data.prepare_dataset(
        exp_path = exp_path,
        dataset = DATASET,
        test_size = TEST_SIZE,
        force = False)

    # plot medoids    
    plots.plot_medoids(exp_path=exp_path)

    # PREPARE THE TASKS
    ###########################################

    # prepare main task
    main_task = ("main", 
        data.prepare_classification_data(
            exp_path = exp_path,
            task_name = "main",
            sts_length = MAIN_STS_LENGTH,
            label_type = "original",
            label_shft = 0,
            batch_size = BATCH_SIZE,
            wndow_size = WINDOW_SIZE,
            force = False))

    plots.plot_OESM(
        exp_path=exp_path, 
        task_name="main",
        sts_range=(140,1400),
        n_patterns=3)

    # # prepare auxiliary tasks
    # aux_tasks = []
    # for i in SHIFTS:
    #     auxt = (f"shift_{i}",
    #         data.prepare_classification_data(
    #             exp_path = exp_path,
    #             task_name = "shift",
    #             sts_length = AUX_STS_LENGTH,
    #             label_type = "discrete_STS",
    #             label_shft = -i,
    #             batch_size = BATCH_SIZE,
    #             wndow_size = WINDOW_SIZE,
    #             force = False))
    #     aux_tasks.append(auxt)

    # # CREATE THE MODELS
    # ###########################################

    # ref_size = main_task[1].ds_train.OESM.shape[1]
    # channels = main_task[1].channels
    # nlabels = main_task[1].labels_size

    # # create the model
    # default_model = model_wrapper(model_architecture=CNN_DTW,
    #     ref_size=ref_size,channels=channels, labels=nlabels,
    #     window_size=WINDOW_SIZE, lr=LEARNING_RATE)
    # pretrain_model = deepcopy(default_model)

    # # RUN THE SEQUENCES
    # ###########################################

    # # run the default sequence
    # default_model_path = train.run_sequence(
    #     exp_path = exp_path,
    #     seq_name = "default",
    #     main_task = main_task,
    #     aux_tasks = [],
    #     model = default_model)

    #  # run the pretrain sequence
    # pretrain_model_path = train.run_sequence(
    #     exp_path = exp_path,
    #     seq_name = "pretrain",
    #     main_task = main_task,
    #     aux_tasks = aux_tasks,
    #     model = pretrain_model)
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
