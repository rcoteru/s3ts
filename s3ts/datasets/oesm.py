from scipy.spatial import distance_matrix
from tqdm import tqdm
import numpy as np

from functools import partial
import multiprocessing as mp
import logging

log = logging.Logger(__name__)

class OES_batch:
    
    """
    On-line Elastic Similarity class

    Parameters
    ----------
    R : 1-D array_like
        time series pattern
    w : float
        On-line similarity measure memory. Must be between 0 and 1.
    rateX : float, optional, default: 1
        Reference Time series generate_files rate
    rateY : float, optional, default: None
        Query Time series generate_files rate. If rateY = None it takes rateX value
    dist : string, optional, default: 'euclidean'
        cost function
        OPTIONS:
        'euclidean': np.sqrt((x_i - y_j)**2)
        'edr':       1 if abs(x_i - y_j) >= epsilon else 0
        'edit':      1 if x_i != y_j else 0
        'erp':       abs(x_i - y_j) if abs(x_i - y_j) >= epsilon else 0
    epsilon : float, optional, default: None
        edr threshold parameter
    """

    def __init__(self, R, w, dist='euclidean', epsilon=None):

        # Check if distance metric choice is valid
        if dist in ['euclidean', 'edr', 'erp', 'edit', 'euclidean_new']:
            self.dist = dist
        else:
            raise NotImplementedError('Incorrect distance metric')

        if isinstance(R, (np.ndarray)) and R.size > 2:
            self.R = R

        if (w >= 0) and (w <= 1):
            self.w = w
        else:
            raise ValueError('w must be between 0 and 1')

        if (epsilon is None) and (dist in ['edr', 'erp']):
            raise ValueError(
                'for dist edr or erp epsilon must be a non negative float')
        elif (epsilon is not None) and epsilon < 0:
            raise ValueError('epsilon must be a non negative float')
        self.epsilon = epsilon

    def init_dist(self, S):
        """
        Initial Similarity Measure

        Parameters
        ----------
        S : 1-D array_like
            Array containing time series observations

        Returns
        -------
        dist: float
            Elastic Similarity Measure between R (pattern time series) and S (query time series)
        """

        if isinstance(S, (np.ndarray)) and S.size > 2:
            pass
        else:
            raise ValueError('S time series must have more than 2 instances')

        # Compute point-wise distance
        if self.dist == 'euclidean':
            RS = self.__euclidean(self.R, S)
        if self.dist == 'euclidean_new':
            RS = self.__euclidean_new(self.R, S)
        elif self.dist == 'edr':
            RS = self.__edr(self.R, S, self.epsilon)
        elif self.dist == 'erp':
            RS = self.__erp(self.R, S, self.epsilon)
        elif self.dist == 'edit':
            RS = self.__edit(self.R, S)

        # compute recursive distance matrix using dynamic programing
        r, s = np.shape(RS)

        # Solve first row
        for j in range(1, s):
            RS[0, j] += self.w * RS[0, j - 1]

        # Solve first column
        for i in range(1, r):
            RS[i, 0] += RS[i - 1, 0]

        # Solve the rest
        for i in range(1, r):
            for j in range(1, s):
                RS[i, j] += np.min([RS[i - 1, j], self.w *
                                    RS[i - 1, j - 1], self.w * RS[i, j - 1]])

        # save statistics
        self.dtwR = RS[:, -1]

        return RS

    def update_dist(self, Y):
        '''
        Add new observations to query time series

        Parameters
        ----------
        Y : array_like or float
            new query time series observation(s)

        Returns
        -------
        dist : float
            Updated distance

        Warning: the time series have to be non-empty
        (at least composed by a single measure)


          -----------
        R | RS | RY |
          -----------
            S    Y
        '''

        if isinstance(Y, (np.ndarray)):
            pass
        else:
            Y = np.array([Y])

        # Solve RY
        dtwRY = self._solveRY(Y, self.dtwR)

        # Save statistics
        self.dtwR = dtwRY

        return dtwRY

    def __euclidean(self, X, Y):
        return distance_matrix(X, Y, p=2)**2

    def __edr(self, X, Y, epsilon):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = 1.0 * (abs(X_tmp - Y_tmp) < epsilon)
        return XY

    def __erp(self, X, Y, epsilon):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = abs(X_tmp - Y_tmp)
        XY[XY < epsilon] = 0
        return XY

    def __edit(self, X, Y):
        Y_tmp, X_tmp = np.meshgrid(Y, X)
        XY = 1.0 * (X_tmp == Y_tmp)
        return XY

    def _solveRY(self, Y, dtwR):
        '''
        R, Y: to (partial) time series
         -----------
        R|prev| RY |
         -----------
           S    Y

        dtwR: partial solutions of DTW(R,S)
        iniI: Index of the first point of R in the complete time series
        iniJ: Index of the first point of Y in the complete time series

        * Warning *: R and Y have to be nonempty (partial) series
        '''

        # Compute point-wise distance
        if self.dist == 'euclidean':
            RY = self.__euclidean(self.R, Y)
        if self.dist == 'euclidean_new':
            RY = self.__euclidean_new(self.R, Y)
        elif self.dist == 'edr':
            RY = self.__edr(self.R, Y, self.epsilon)
        elif self.dist == 'erp':
            RY = self.__erp(self.R, Y, self.epsilon)
        elif self.dist == 'edit':
            RY = self.__edit(self.R, Y)

        r, n = np.shape(RY)

        # First first column
        RY[0, 0] += self.w * dtwR[0]
        for i in range(1, r):
            RY[i, 0] += np.min([
                RY[i - 1, 0], self.w * dtwR[i - 1], self.w * dtwR[i]])

        # Solve first row
        for j in range(1, n):
            RY[0, j] += self.w * RY[0, j - 1]

        # Solve the rest
        for j in range(1, n):
            for i in range(1, r):
                RY[i, j] += np.min([RY[i - 1, j], self.w *
                                    RY[i - 1, j - 1], self.w * RY[i, j - 1]])

        return RY[:, -1]


def compute_odtw_distance_matrix(pattern, sts, rho, dist="euclidean"):
    
    """
    Compute distance matrix.
    :param ref: reference pattern
    :param stream: stream time series
    :param rho: memory
    :return: distance matrices
    """

    # print('Computing ODTW distance matrix')

    distMat = np.zeros((ref.shape[0], stream.shape[0]))
    
    odtw = OEM_batch(ref, rho, dist=dist)
    distMat[:, :3] = odtw.init_dist(stream[:3])

    for point in range(3, stream.shape[0]):
        if type(stream[point]) is list or type(stream[point]) is np.ndarray:
            tmp_stream = np.expand_dims(stream[point], axis=0)
        else:
            tmp_stream = [stream[point]]
        distMat[:, point] = odtw.update_dist(tmp_stream)

    return distMat


def compute_ESM_sc(
        sts_idx: np.ndarray, 
        STSs: np.ndarray, 
        patterns: np.ndarray, 
        rho: float,
        lock: mp.Lock, 
    ):

    sts_length = STSs.shape[1]
    n_patts = patterns.shape[0]
    patt_length = patterns.shape[1]

    # Initialize progress bar
    with lock:
        bar = tqdm(
            desc=f'STS_{sts_idx}',
            total=n_patts,
            position=sts_idx,
            leave=True
        )

    partial_ESM = np.zeros((n_patts, patt_length, sts_length))

    for patt_idx in range(n_patts):
        partial_ESM[patt_idx, :, :] = compute_odtw_distance_matrix(
            pattern=patterns[patt_idx], sts=STSs[sts_idx], 
            rho=rho, dist="euclidean_new")
        with lock: # Update the progress bar
            bar.update(1)

    with lock:  # Close progress bar
        bar.close()

    return partial_ESM

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def compute_ESMs(
        STSs: np.ndarray, 
        patterns: np.ndarray,
        rho: float,
        nprocs: int = 4
    ):

    assert(len(STSs) == 2)
    n_STS = STSs.shape[1]

    assert(len(patterns.shape) == 2)
    n_patts = patterns.shape[0]
    l_samp = patterns.shape[1]

    scaled_rho = rho ** (1 / l_samp)

    lock = mp.Manager().Lock()

    # IDs to send to each process
    STS_ids = np.arange(n_STS).tolist()

    # INFO: "partial" basically makes "wrapper_compute" take only the data as argument
    compute_ESM_sc_call = partial(compute_ESM_sc, STSs=STSs,
                                patterns=patterns, rho=scaled_rho,
                                lock=lock)

    with mp.Pool(processes=nprocs) as pool:
        full_DTWs = pool.map(compute_ESM_sc_call, STS_ids)

    full_DTWs = np.array(full_DTWs)