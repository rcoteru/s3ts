# sktime
from sktime.clustering.k_medoids import TimeSeriesKMedoids

# numpy / scipy
from scipy.spatial import distance_matrix
import numpy as np

from s3ts.structures import AugProbabilities
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
            
            sample = X_train[idx,:].copy()
            label = Y_train[idx]
            
            r = r + nsamples_train
            STS_X[r*sample_length:(r+1)*sample_length] = sample
            STS_Y[r*sample_length:(r+1)*sample_length] = label

    else:
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

    return STS_X, STS_Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #