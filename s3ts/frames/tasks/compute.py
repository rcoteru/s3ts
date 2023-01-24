# sktime
from sktime.clustering.k_medoids import TimeSeriesKMedoids

# numpy / scipy
from scipy.spatial import distance_matrix
from math import ceil
import numpy as np
import logging


log = logging.Logger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_medoids(
        X: np.ndarray, Y: np.ndarray,
        distance_type: str = 'euclidean'
    ) -> tuple[np.ndarray, np.ndarray]: 

    """ Computes the medoids of the classes in the dataset. """

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_STS(
        X: np.ndarray, 
        Y: np.ndarray,
        target_nframes: int,
        frame_buffer: int = 0,
        random_state: int = 0,
        ) -> tuple[np.ndarray, np.ndarray]:

    """
    Builds an STS from and array of samples and labels.
    """

    assert(len(X.shape) == 2)
    assert(X.shape[0] == Y.shape[0])
    rng = np.random.default_rng(seed=random_state)
    
    nsamples = X.shape[0]
    l_sample = X.shape[1]
    
    # recommended number of frames
    rec_nframes = nsamples*l_sample

    print("Sample size:", target_nframes)
    print("Number of samples:", nsamples)
    print("Target number of frames:", target_nframes)
    print("Recom. number of frames:", rec_nframes)

    if target_nframes < rec_nframes:
        print(f"WARNING: Target number of frames {target_nframes} below"
                f"recommended {rec_nframes} for {nsamples} of size {l_sample}")

    target_nsamples = ceil((target_nframes + frame_buffer)/float(l_sample))

    STS_X = np.empty(target_nsamples*l_sample)
    STS_Y = np.empty(target_nsamples*l_sample)

    for r in range(target_nsamples):
        random_idx = rng.integers(0, nsamples)
        STS_X[r*l_sample:(r+1)*l_sample] = X[random_idx,:]
        STS_Y[r*l_sample:(r+1)*l_sample] = Y[random_idx]
        
    return STS_X, STS_Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_STS_old(
        X_train: np.ndarray, 
        Y_train: np.ndarray, 
        X_test: np.ndarray,
        Y_test: np.ndarray,
        random_state: int = 0,
        nsamples_sts: int = None,
        skip_ids: list[int] = []
        ) -> tuple[np.ndarray, np.ndarray]:

    """
    Builds an STS from and array of samples and labels.

    If sts_length is an int, builds it from randomly picked samples.
    If sts_length is None, builds it with a randomly sampled permutation of them. 
    
    """

    assert(X_train.shape[0] == Y_train.shape[0])
    nsamples_train = X_train.shape[0]
    nsamples_test = X_test.shape[0]
    nsamples = nsamples_train + nsamples_test

    assert(len(X_train.shape) == 2)
    sample_length = X_train.shape[1]

    rng = np.random.default_rng(seed=random_state)

    if nsamples_sts is None:
        
        STS_X = np.empty(nsamples*sample_length)
        STS_Y = np.empty(nsamples*sample_length)

        # train samples
        for r, idx in enumerate(rng.permutation(np.arange(nsamples_train))):
            
            sample = X_train[idx,:].copy()
            label = Y_train[idx]

            STS_X[r*sample_length:(r+1)*sample_length] = sample
            STS_Y[r*sample_length:(r+1)*sample_length] = label
        
        # test samples
        for r, idx in enumerate(rng.permutation(np.arange(nsamples_test))):
            
            sample = X_test[idx,:].copy()
            label = Y_test[idx]
            
            r = r + nsamples_train
            STS_X[r*sample_length:(r+1)*sample_length] = sample
            STS_Y[r*sample_length:(r+1)*sample_length] = label

    else: # TODO fix, broken
        STS_X = np.empty(nsamples_sts*sample_length)
        STS_Y = np.empty(nsamples_sts*sample_length)

        for r in range(nsamples_sts):
            while True:
                random_idx = rng.integers(0, nsamples)
                if random_idx in skip_ids:
                    continue
                else:
                    break

            STS_X[r*sample_length:(r+1)*sample_length] = X[random_idx,:]
            STS_Y[r*sample_length:(r+1)*sample_length] = Y[random_idx]

    return STS_X, STS_Y, nsamples_test/nsamples