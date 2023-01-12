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

def build_STS(
        X: np.ndarray, 
        Y: np.ndarray, 
        sts_length: int,
        buffer_length: int,
        skip_ids: list[int] = [],
        aug_probs: AugProbabilities = None,
        random_state: int = 0,
        ) -> tuple[np.ndarray, np.ndarray]:

    """
    Builds an STS from and array of samples and labels.

    If sts_length is an int, builds it from randomly picked samples.
    If sts_length is None, builds it with a randomly sampled permutation of them. 
    
    """

    assert(X.shape[0] == Y.shape[0])
    nsamples = X.shape[0]

    assert(len(X.shape) == 2)
    s_length = X.shape[1]

    if sts_length is None:
        STS_X = np.empty((buffer_length + nsamples)*s_length)
        STS_Y = np.empty((buffer_length + nsamples)*s_length)
    else:
        STS_X = np.empty((buffer_length + sts_length)*s_length)
        STS_Y = np.empty((buffer_length + sts_length)*s_length)

    # TODO implement augmentations
    def augment(sample: np.ndarray):
        if rng.random() <= aug_probs.jitter:
            sample = sample
        if rng.random() <= aug_probs.scaling:
            sample = sample
        if rng.random() <= aug_probs.time_warp:
            sample = sample
        if rng.random() <= aug_probs.window_warp:
            sample = sample

    rng = np.random.default_rng(seed=random_state)

    if sts_length is None:
        random_fill = buffer_length
    else:
        random_fill = buffer_length + sts_length

    # fill with randomly picked smples
    for r in range(random_fill):
        while True:
            random_idx = rng.integers(0, nsamples)
            if random_idx in skip_ids:
                continue
            else:
                break

        sample = X[random_idx,:].copy()
        label = Y[random_idx]

        if aug_probs is not None:
            sample = augment(sample)
        
        STS_X[r*s_length:(r+1)*s_length] = sample
        STS_Y[r*s_length:(r+1)*s_length] = label

    # fill with random permutation of samples
    if sts_length is None:

        for r, idx in enumerate(rng.permutation(np.arange(nsamples))):

            r = r + random_fill
            sample = X[idx,:].copy()
            label = Y[idx]

            if aug_probs is not None:
                sample = augment(sample)

            STS_X[r*s_length:(r+1)*s_length] = sample
            STS_Y[r*s_length:(r+1)*s_length] = label

    return STS_X, STS_Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_OESM():


    pass