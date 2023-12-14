from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def minmaxscale(X: np.ndarray, fmin: float = -1, fmax: float =1) -> np.ndarray:

    """ Scales the input array to the range [-1, 1]. """

    Xmin, Xmax = X[0], X[0]
    for i in range(X.shape[0]):
        if X[i] < Xmin:
            Xmin = X[i]
        if X[i] > Xmax:
            Xmax = X[i]

    X_std = (X - Xmin) / (Xmax - Xmin)
    X_scaled = X_std * (fmax - fmin) + fmin

    return X_scaled


@jit(nopython=True, parallel=True)
def compute_GM_optim(STS: np.ndarray, 
                    patterns: np.ndarray,
                    ) -> np.ndarray:

    """ Computes the gramian matrix (GM) for a given set of patterns and a given STS.
        Optimized version using Numba.
        
        The GM has dimensions (n_patts, l_patts, STS_length), where n_patts is the number of patterns,
        l_patts is the length of the patterns, and STS_length is the length of the STS.
        
        Parameters
        ----------
        STS : np.ndarray
            The STS to compute the GM for.
        patterns : np.ndarray
            The patterns used to compute the GM.

        References
    ----------
    .. [1] Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
           Inspection and Classification Using Tiled Convolutional Neural
           Networks." AAAI Workshop (2015).
    """

    feature_range: tuple[int] = (-1, 1)

    n_patts: int = patterns.shape[0]
    l_patts: int = patterns.shape[1]
    l_STS: int = STS.shape[0]

    STS_cos = minmaxscale(STS, fmin=feature_range[0], fmax=feature_range[1])
    STS_sin = np.sqrt(np.clip(1 - STS_cos ** 2, 0, 1))

    patterns_cos = np.zeros_like(patterns)
    for i in prange(n_patts):
        patterns_cos[i,:] = minmaxscale(patterns[i,:], fmin=feature_range[0], fmax=feature_range[1])
    patterns_sin = np.sqrt(np.clip(1 - patterns_cos ** 2, 0, 1))

    # Compute the Gramian summantion distance matrix
    GM = np.empty((n_patts, l_patts, l_STS), dtype=np.float32)
    for p in prange(n_patts):
        for i in prange(l_STS):
            for j in prange(l_patts):
                GM[p, j, i] = STS_cos[i]*patterns_cos[p, j] + STS_sin[i]*patterns_sin[p, j]

    # Return the full distance matrix
    return GM


if __name__ == "__main__":


    """ Small test script for optimization. """

    import matplotlib.pyplot as plt 

    STS = np.sin(np.linspace(0, 6*np.pi, 10000))
    lpat = 300
    patterns = np.stack([np.arange(0, lpat), np.arange(0, lpat)[::-1]])
    # standardize patterns
    patterns = (patterns - np.mean(patterns, axis=1, keepdims=True)) / np.std(patterns, axis=1, keepdims=True)
    #print(patterns)

    GM = compute_GM_optim(STS, STS.reshape(1, -1))

    plt.figure()
    plt.plot(STS)

    plt.figure()
    plt.imshow(GM[0])

    print(GM.max(), GM.min())
    
    plt.show()


