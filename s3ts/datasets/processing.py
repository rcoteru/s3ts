from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.datasets import load_UCR_UEA_dataset

from sklearn.preprocessing import KBinsDiscretizer

from scipy.spatial import distance_matrix
import numpy as np

import logging

from s3ts import RANDOM_STATE
rng = np.random.default_rng(seed=RANDOM_STATE)
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
        Y = Y.astype(int)
        Y = Y - Y.min()
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

def build_STS(
        X: np.ndarray, 
        Y: np.ndarray, 
        sts_length: int,
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

    STS_X = np.empty(sts_length*s_length)
    STS_Y = np.empty(sts_length*s_length)

    for r in range(sts_length):

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

        STS_X[r*s_length:(r+1)*s_length] = sample
        STS_Y[r*s_length:(r+1)*s_length] = label

    return STS_X, STS_Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def discretize_TS(
        TS: np.ndarray,
        intervals: int, 
        strategy: str = "quantile", # uniform (width) / quantile (freq) 
        random_state: int = 0
        ) -> np.ndarray:

    kbd = KBinsDiscretizer(n_bins=intervals, encode="ordinal",
            strategy=strategy, random_state=random_state)
    kbd.fit(TS)
    
    return kbd.transform(TS), kbd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def shift_labels(
        labels: np.ndarray,
        OESM: np.ndarray,
        STS: np.ndarray,
        shift: int, 
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if shift == 0:
        return labels, OESM, STS

    sts_length = labels.shape[0]
    STS_d = np.copy(STS)
    OESM_d = np.copy(OESM)

    if shift < 0:

        labels_d = np.roll(labels, shift, axis=0).copy()
        labels_d = np.delete(labels_d, np.arange(sts_length+shift, sts_length), axis=0)
        OESM_d = np.delete(OESM_d, np.arange(sts_length+shift, sts_length), axis=2)
        STS_d = np.delete(STS_d, np.arange(sts_length+shift, sts_length), axis=0)

    elif shift > 0:

        labels_d = np.roll(labels, shift, axis=0).copy()
        labels_d = np.delete(labels_d, np.arange(shift), axis=0)
        OESM_d = np.delete(OESM_d, np.arange(shift), axis=2)
        STS_d = np.delete(STS_d, np.arange(shift), axis=0)

    else: 
        raise NotImplementedError()

    return labels_d, OESM_d, STS_d
        
    