from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.datasets import load_UCR_UEA_dataset

from sklearn.preprocessing import KBinsDiscretizer

from scipy.spatial import distance_matrix
import numpy as np

import logging

rng = np.random.default_rng(seed=0)
log = logging.Logger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def acquire_dataset(dataset_name: str) -> None:

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
        float(Y[0])
        mapping = None
    except ValueError:
        mapping = {k: v for v, k in enumerate(np.unique(Y))}
        
    return X, Y, mapping

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_medoids(
        X: np.ndarray, Y: np.ndarray,
        distance_type: str = 'euclidean'
    ) -> tuple[np.ndarray, np.ndarray]: 

    assert(len(X.shape) == 2)
    s_length = X.shape[1]

    nclasses = len(np.unique(Y))

    # Find the medoids for each class
    medoids = np.empty((nclasses, s_length), dtype=float)
    medoid_ids = np.empty(nclasses, dtype=int)
    for i, y in enumerate(np.unique(Y)):

        index = np.argwhere(Y == y)
        Xi = X[index, :]

        # ...using simple euclidean distance        
        if distance_type == "euclidean":
            medoid_idx = np.argmin(distance_matrix(Xi.squeeze(), Xi.squeeze()).sum(axis=0))
            medoids[i,:] = Xi[medoid_idx,:]
            medoid_ids[i] = index[medoid_idx]

        # ...using Dynamic Time Warping (DTW)
        elif distance_type == "dtw":
            tskm = TimeSeriesKMedoids(n_clusters=1, init_algorithm="forgy", metric="dtw")
            tskm.fit(Xi)
            medoids[i,:] = tskm.cluster_centers_.squeeze()
            medoid_ids[i] = np.where(np.all(Xi.squeeze() == medoids[i,:], axis=1))[0][0]

        else:
            raise NotImplementedError

    return medoids, medoid_ids

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def build_STSs(
        X: np.ndarray, 
        Y: np.ndarray, 
        samples_per_sts: int,
        number_of_sts: int = 1,
        skip_ids: list[int] = [],
        aug_jitter: float = 0,
        aug_scaling: float = 0,
        aug_time_warp: float = 0,
        aug_window_warp: float = 0,
        ) -> tuple[np.ndarray, np.ndarray]:

    assert(X.shape[0] == Y.shape[0])
    nsamples = X.shape[0]

    assert(len(X.shape) == 2)
    s_length = X.shape[1]
    sts_length = samples_per_sts*s_length

    STSs_X = np.empty((number_of_sts, sts_length))
    STSs_Y = np.empty((number_of_sts,sts_length))

    for nsts in range(number_of_sts):
        for r in range(samples_per_sts):

            while True:
                random_idx = rng.integers(0, nsamples)
                if random_idx in skip_ids:
                    continue
                else:
                    break

            sample = X[random_idx,:].copy()
            label = Y[random_idx]

            # TODO implement augmentations
            if rng.random() <= aug_jitter:
                sample = sample
            if rng.random() <= aug_scaling:
                sample = sample
            if rng.random() <= aug_time_warp:
                sample = sample
            if rng.random() <= aug_window_warp:
                sample = sample

            STSs_X[nsts, r*s_length:(r+1)*s_length] = sample
            STSs_Y[nsts, r*s_length:(r+1)*s_length] = label

    return STSs_X, STSs_Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def discretize_STSs(
        STSs: np.ndarray,
        intervals: int, 
        strategy: str = "quantile", # uniform (width) / quantile (freq) 
        random_state: int = 0
        ) -> np.ndarray:

    kbd = KBinsDiscretizer(n_bins=intervals, encode="ordinal",
            strategy=strategy, random_state=random_state)

    kbd.fit(STSs)
    
    return kbd.transform(STSs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def shift_STSs_labels(
        STSs_X: np.ndarray,
        STSs_Y: np.ndarray,
        shift: int, 
        ) -> tuple[np.ndarray, np.ndarray]:

    sts_length = STSs_X.shape[1]
    STSs_Xd = np.copy(STSs_X)

    if shift < 0:
        STSs_Yd = np.roll(STSs_Y, shift, axis=1).copy()
        STSs_Xd = np.delete(STSs_Xd, np.arange(sts_length+shift, sts_length), axis=1)
        STSs_Yd = np.delete(STSs_Yd, np.arange(sts_length+shift, sts_length), axis=1)
    elif shift > 0:
        STSs_Yd = np.roll(STSs_Y, shift, axis=1).copy()
        STSs_Xd = np.delete(STSs_Xd, np.arange(shift), axis=1)
        STSs_Yd = np.delete(STSs_Yd, np.arange(shift), axis=1)
    else: 
        STSs_Yd = np.copy(STSs_Y)
    
    return STSs_Xd, STSs_Yd