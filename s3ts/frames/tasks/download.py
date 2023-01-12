from sktime.datasets import load_UCR_UEA_dataset
import numpy as np 

import logging

log = logging.Logger(__name__)


def download_dataset(dataset_name: str) -> None:

    log.info("DOWNLOADING UCR/UEA DATASET")
    log.info("Dataset name:", dataset_name)

    # Download TS dataset from UCR UEA
    X, Y = load_UCR_UEA_dataset(name=dataset_name, 
                            return_type="np2d",
                            return_X_y=True)

    nsamples = X.shape[0]
    s_length = X.shape[1]
    nclasses = len(np.unique(Y))

    log.info("Number of samples:", nsamples)
    log.info("Sample length:", s_length)
    log.info("Number of classes:", nclasses)
    log.info("DOWNLOAD FINISHED")

    try:
        Y = Y.astype(int)
        Y = Y - Y.min()
        mapping = None
    except ValueError:
        mapping = {k: v for v, k in enumerate(np.unique(Y))}
        
    return X, Y, mapping