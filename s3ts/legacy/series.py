# data processing stuff
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from scipy.spatial import distance_matrix
import logging as log
import numpy as np

def compute_medoids(
        X: np.ndarray, 
        Y: np.ndarray,
        distance_type: str = 'dtw'
    ) -> tuple[np.ndarray, np.ndarray]: 

    """ Computes the medoids of the classes in the dataset. 
    
    Parameters
    ----------
    X : np.ndarray
        The time series dataset.
    Y : np.ndarray
        The labels of the time series dataset.
    distance_type : str, optional
        The distance type to use, by default 'dtw'
    """

    # Check the distance type
    if distance_type not in ["euclidean", "dtw"]:
        raise ValueError("The distance type must be either 'euclidean' or 'dtw'.")
    
    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of events in the dataset and labels must be the same.")

    # Get the number of classes
    n_classes = len(np.unique(Y))
    
    # Get the length of the time series
    s_length = X.shape[1]

    # Initialize the arrays
    medoids = np.empty((n_classes, s_length), dtype=float)
    medoid_ids = np.empty(n_classes, dtype=int)
    
    # Find the medoids for each class
    for i, y in enumerate(np.unique(Y)):

        # Get the events of the class
        index = np.argwhere(Y == y)
        Xy = X[index, :]

        # ...using Euclidean distance        
        if distance_type == "euclidean":
            medoid_idx = np.argmin(distance_matrix(Xy.squeeze(), Xy.squeeze()).sum(axis=0))
            medoids[i,:] = Xy[medoid_idx,:]
            medoid_ids[i] = index[medoid_idx]

        # ...using Dynamic Time Warping (DTW)
        if distance_type == "dtw":
            if Xy.shape[0] > 1:
                tskm = TimeSeriesKMedoids(n_clusters=1, init_algorithm="forgy", metric="dtw")
                tskm.fit(Xy)
                medoids[i,:] = tskm.cluster_centers_.squeeze()
                medoid_ids[i] = np.where(np.all(Xy.squeeze() == medoids[i,:], axis=1))[0][0]
            else:
                medoids[i,:] = Xy.squeeze()
                medoid_ids[i] = np.where(np.all(Xy.squeeze() == medoids[i,:], axis=1))[0][0]

    # Return the medoids and their indices
    return medoids, medoid_ids

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_STS(
        X: np.ndarray, 
        Y: np.ndarray,
        STS_events: int,
        shift_limits: bool,
        mode: str = "random",
        random_state: int = 0,
        event_strat_size: int = 2,
        add_first_event: bool = False,
        ) -> tuple[np.ndarray, np.ndarray]:

    """ Generates a Streaming Time Series (STS) from a given dataset. """

    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of events in the dataset and labels must be the same.")

    # Set the random state for reproducibility
    rng = np.random.default_rng(seed=random_state)
    
    # Get the number of classes
    n_classes = len(np.unique(Y))
    
    # Get the length of the time series
    s_length = X.shape[1]
    
    # Get the number of events
    n_events = X.shape[0]

    # Get the length of the final STS
    STS_length = STS_events*s_length

    # Do some logging
    log.info(f"Number of events: {n_events}")
    log.info(f"Length of events: {s_length}")
    log.info(f"Number of classes: {n_classes}")
    log.info(f"Class ratios: {np.unique(Y, return_counts=True)[1]/n_events}")
    log.info(f"Length of STS: {STS_length}")

    # Initialize the arrays
    if add_first_event:
        STS = np.empty(STS_length+s_length, dtype=np.float32)
        SCS = np.empty(STS_length+s_length, dtype=np.int8)
        random_idx = rng.integers(0, n_events)
        STS[0:s_length] = X[random_idx,:]
        SCS[0:s_length] = Y[random_idx]
    else:
        STS = np.empty(STS_length, dtype=np.float32)
        SCS = np.empty(STS_length, dtype=np.int8)

    # Generate the STS 
    if mode == "random":
        for s in range(STS_events):

            random_idx = rng.integers(0, n_events)
            s = s+1 if add_first_event else s

            # Calculate shift so that sample ends match
            shift = STS[s-1] - X[random_idx,0] if shift_limits else 0

            STS[s*s_length:(s+1)*s_length] = X[random_idx,:] + shift
            SCS[s*s_length:(s+1)*s_length] = Y[random_idx]

    if mode == "stratified":
        
        exc =  n_events//n_classes

        if exc%event_strat_size != 0:
            raise ValueError("The number of events per class must be a multiple of the event stratification size.")
    
        if STS_events%exc != 0:
            raise ValueError("The number of events in the STS must be a multiple of the number of events per class.")

        event_idx = np.arange(X.shape[0])
        
        clist = []
        for c in np.unique(Y):
            Yc_idx = event_idx[Y==c]
            rng.shuffle(Yc_idx)
            clist.append(np.reshape(Yc_idx, (-1, event_strat_size)))

        strats = np.concatenate(clist, axis=1)
        n_repeats = STS_events // n_events

        cidx = 1 if add_first_event else 0
        for strat in range(strats.shape[0]):
            for _ in range(n_repeats):
                for s in rng.permutation(strats[strat,:]):

                    # Calculate shift so that sample ends match
                    shift = STS[cidx-1] - X[s,0] if shift_limits else 0

                    STS[cidx*s_length:(cidx+1)*s_length] = X[s,:] + shift
                    SCS[cidx*s_length:(cidx+1)*s_length] = Y[s]

                    # Calculate shift so that sample ends match
                    cidx += 1

    # Normalize the STS
    STS = (STS - np.mean(STS))/np.std(STS)

    # Return the STS and the SCS
    return STS, SCS