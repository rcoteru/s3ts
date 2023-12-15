from aeon.distances import pairwise_distance
import numpy as np

def compute_medoids(
        X: np.ndarray, Y: np.ndarray, 
        meds_per_class: int = 1, metric: str = 'dtw', 
    ) -> tuple[np.ndarray, np.ndarray]: 

    """ Computes 'meds_per_class' medoids of each class in the dataset. """

    # Check the distance type
    suported_metrics = ['euclidean', 'squared',
        'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp', 'msm']
    if metric not in suported_metrics:
        raise ValueError(f"The distance type must be one of {suported_metrics}.")
    
    # grab the classes
    sdim, slen = X.shape[1], X.shape[2]
    classes = np.unique(Y)

    # Initialize the arrays
    meds = np.empty((len(np.unique(Y)), meds_per_class, sdim, slen), dtype=float)
    meds_idx = np.empty((len(np.unique(Y)), meds_per_class), dtype=int)
    
    # Find the medoids for each class
    for i, y in enumerate(classes):
        index = np.argwhere(Y == y)[:,0]
        X_y = X[index,:,:]
        dm = pairwise_distance(X_y, metric=metric)
        scores = dm.sum(axis=0)
        meds_idx_y = np.argpartition(scores, meds_per_class)[:meds_per_class]
        print(X_y[meds_idx_y].shape)
        meds[i,:,:,:] = X_y[meds_idx_y]
        meds_idx[i,:] = index[meds_idx_y]

    # Return the medoids and their indices
    return meds, meds_idx

# Random event probability
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def fin_random_STS(
        X: np.ndarray, Y: np.ndarray, length: int, seed: int = 42
        ) -> tuple[np.ndarray, np.ndarray]:
    
    """ Create a finite STS by randomly concatenating 'length' events """

    # create random number generator
    rng = np.random.default_rng(seed=seed)

    # get dataset info
    nsamp, lsamp = X.shape[0], X.shape[2]
    STS_dim, STS_len = X.shape[1], X.shape[2]*length

    # initialize STS
    STS = np.zeros((STS_dim, STS_len), dtype=np.float32)
    SCS = np.zeros(STS_len, dtype=np.int8)

    # random concatenation
    for s in range(length):
        rand_idx = rng.integers(0, nsamp)
        STS[:, s*lsamp:(s+1)*lsamp] = X[rand_idx,:,:] 
        SCS[ s*lsamp : (s+1)*lsamp ] = Y[rand_idx]

    return STS, SCS

def inf_random_STS(
        X: np.ndarray, Y: np.ndarray, seed: int = 42
        ) -> tuple[np.ndarray, int]: # type: ignore
    
    # create random number generator
    rng = np.random.default_rng(seed=seed)

    # get dataset info
    nsamp, lsamp = X.shape[0], X.shape[2]

    # random concatenation
    while True:
        rand_idx = rng.integers(0, nsamp)
        for i in range(lsamp):
            yield X[rand_idx,:,i], Y[rand_idx] # type: ignore

# Balanced event probability
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# TODO actualy implement the balanced sampling

def fin_balanced_STS(
        X: np.ndarray, Y: np.ndarray, length: int, seed: int = 42
        ) -> tuple[np.ndarray, int]:
    
    """ Create a finite STS by randomly concatenating 'length' events """

    # create random number generator
    rng = np.random.default_rng(seed=seed)

    # get dataset info
    nsamp, lsamp = X.shape[0], X.shape[2]
    STS_dim, STS_len = X.shape[1], X.shape[2]*length

    # initialize STS
    STS = np.zeros((STS_dim, STS_len), dtype=np.float32)
    SCS = np.zeros(STS_len, dtype=np.int8)

    # random concatenation
    for s in range(length):
        rand_idx = rng.integers(0, nsamp)
        STS[:, s*lsamp:(s+1)*lsamp] = X[rand_idx,:,:] 
        SCS[ s*lsamp : (s+1)*lsamp ] = Y[rand_idx]

    return STS, SCS

def inf_balanced_STS(
        X: np.ndarray, Y: np.ndarray, seed: int = 42
        ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    
    # create random number generator
    rng = np.random.default_rng(seed=seed)

    # get dataset info
    nsamp, lsamp = X.shape[0], X.shape[2]

    # random concatenation
    while True:
        rand_idx = rng.integers(0, nsamp)
        for i in range(lsamp):
            yield X[rand_idx,:,i], Y[rand_idx] # type: ignore




























# def compute_STS(
#         X: np.ndarray, 
#         Y: np.ndarray,
#         STS_events: int,
#         shift_limits: bool,
#         mode: str = "random",
#         random_state: int = 0,
#         event_strat_size: int = 2,
#         add_first_event: bool = False,
#         ) -> tuple[np.ndarray, np.ndarray]:

#     """ Generates a Streaming Time Series (STS) from a given dataset. """

#     # Check the shape of the dataset and labels match
#     if X.shape[0] != Y.shape[0]:
#         raise ValueError("The number of events in the dataset and labels must be the same.")

    
    
#     # Get the number of classes
#     n_classes = len(np.unique(Y))
    
#     # Get the length of the time series
#     s_length = X.shape[1]
    
#     # Get the number of events
#     n_events = X.shape[0]

#     # Get the length of the final STS
#     STS_length = STS_events*s_length

#     # Do some logging
#     log.info(f"Number of events: {n_events}")
#     log.info(f"Length of events: {s_length}")
#     log.info(f"Number of classes: {n_classes}")
#     log.info(f"Class ratios: {np.unique(Y, return_counts=True)[1]/n_events}")
#     log.info(f"Length of STS: {STS_length}")

#     # Initialize the arrays
#     if add_first_event:
#         STS = np.empty(STS_length+s_length, dtype=np.float32)
#         SCS = np.empty(STS_length+s_length, dtype=np.int8)
#         random_idx = rng.integers(0, n_events)
#         STS[0:s_length] = X[random_idx,:]
#         SCS[0:s_length] = Y[random_idx]
#     else:
#         STS = np.empty(STS_length, dtype=np.float32)
#         SCS = np.empty(STS_length, dtype=np.int8)

#     # Generate the STS 
#     if mode == "random":
#         for s in range(STS_events):

#             random_idx = rng.integers(0, n_events)
#             s = s+1 if add_first_event else s

#             # Calculate shift so that sample ends match
#             shift = STS[s-1] - X[random_idx,0] if shift_limits else 0

#             STS[s*s_length:(s+1)*s_length] = X[random_idx,:] + shift
#             SCS[s*s_length:(s+1)*s_length] = Y[random_idx]

#     if mode == "stratified":
        
#         exc =  n_events//n_classes

#         if exc%event_strat_size != 0:
#             raise ValueError("The number of events per class must be a multiple of the event stratification size.")
    
#         if STS_events%exc != 0:
#             raise ValueError("The number of events in the STS must be a multiple of the number of events per class.")

#         event_idx = np.arange(X.shape[0])
        
#         clist = []
#         for c in np.unique(Y):
#             Yc_idx = event_idx[Y==c]
#             rng.shuffle(Yc_idx)
#             clist.append(np.reshape(Yc_idx, (-1, event_strat_size)))

#         strats = np.concatenate(clist, axis=1)
#         n_repeats = STS_events // n_events

#         cidx = 1 if add_first_event else 0
#         for strat in range(strats.shape[0]):
#             for _ in range(n_repeats):
#                 for s in rng.permutation(strats[strat,:]):

#                     # Calculate shift so that sample ends match
#                     shift = STS[cidx-1] - X[s,0] if shift_limits else 0

#                     STS[cidx*s_length:(cidx+1)*s_length] = X[s,:] + shift
#                     SCS[cidx*s_length:(cidx+1)*s_length] = Y[s]

#                     # Calculate shift so that sample ends match
#                     cidx += 1

#     # Normalize the STS
#     STS = (STS - np.mean(STS))/np.std(STS)

#     # Return the STS and the SCS
#     return STS, SCS