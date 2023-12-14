from numba import jit, prange
from typing import Optional
from math import sqrt
import numpy as np

# Disimilarity Frames
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@jit(nopython=True, parallel=True)
def compute_DM(
        STS: np.ndarray, 
        patts: np.ndarray, 
        rho: float, 
        metric: str = "euclidean",
        prev_col: Optional[np.ndarray] = None,
        ) -> np.ndarray:
    
    if metric not in ["euclidean", "squared"]:
        raise NotImplementedError

    STS_len: int = STS.shape[1]
    npatts: int = patts.shape[0]
    dpatts: int = patts.shape[1]
    lpatts: int = patts.shape[2]
    w: float = rho**(1/lpatts)

    # Compute point-wise distance
    DM = np.zeros((npatts, lpatts, STS_len), dtype=np.float32)
    if metric == "euclidean":
        for p in prange(npatts):
            for i in prange(lpatts):
                for j in prange(STS_len):
                    for k in prange(dpatts):
                        DM[p, i, j] += (patts[p,k,i] - STS[k,j])**2
                    DM[p, i, j] = sqrt(DM[p, i, j])
    if metric == "squared":
        for p in prange(npatts):
            for i in prange(lpatts):
                for j in prange(STS_len):
                    for k in range(dpatts):
                        DM[p, i, j] += (patts[p,k,i] - STS[k,j])**2

    # Compute the DM
    if prev_col is None:
        for p in prange(npatts):
            # Solve first row
            for j in range(1, STS_len):
                DM[p,0,j] += w*DM[p,0,j-1]
            # Solve first column
            for i in range(1, lpatts):
                DM[p,i,0] += DM[p,i-1,0]
            # Solve the rest
            for i in range(1, lpatts):
                for j in range(1, STS_len):
                    DM[p,i,j] += min([DM[p,i-1,j], w*DM[p,i-1,j-1], w*DM[p,i,j-1]])
    else:
        for p in prange(npatts):
            # First first column
            DM[p,0,0] += w*prev_col[p,0]
            for i in range(1, lpatts):
                DM[p,i,0] += min([
                    DM[p,i-1,0], w*prev_col[p,i-1], w*prev_col[p,i]])
            # Solve first row
            for j in range(1, STS_len):
                DM[p,0,j] += w*DM[p,0,j-1]
            # Solve the rest
            for i in range(1, lpatts):
                for j in range(1, STS_len):
                    DM[p,i,j] += min([
                        DM[p,i-1,j], w*DM[p,i-1,j-1], w*DM[p,i,j-1]])

    # Return the DM
    return DM

# Gramian Frames
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# TODO implement gramian matrix

@jit(nopython=True, parallel=True)
def compute_GM(
        STS: np.ndarray, 
        patts: np.ndarray, 
        rho: float, 
        metric: str = "euclidean",
        prev_col: Optional[np.ndarray] = None,
        ) -> np.ndarray:
    
    if metric not in ["euclidean", "squared"]:
        raise NotImplementedError

    STS_len: int = STS.shape[1]
    npatts: int = patts.shape[0]
    dpatts: int = patts.shape[1]
    lpatts: int = patts.shape[2]
    w: float = rho**(1/lpatts)

    # Compute point-wise distance
    DM = np.zeros((npatts, lpatts, STS_len), dtype=np.float32)
    if metric == "euclidean":
        for p in prange(npatts):
            for i in prange(lpatts):
                for j in prange(STS_len):
                    for k in prange(dpatts):
                        DM[p, i, j] += (patts[p,k,i] - STS[k,j])**2
                    DM[p, i, j] = sqrt(DM[p, i, j])
    if metric == "squared":
        for p in prange(npatts):
            for i in prange(lpatts):
                for j in prange(STS_len):
                    for k in range(dpatts):
                        DM[p, i, j] += (patts[p,k,i] - STS[k,j])**2

    # Compute the DM
    if prev_col is None:
        for p in prange(npatts):
            # Solve first row
            for j in range(1, STS_len):
                DM[p,0,j] += w*DM[p,0,j-1]
            # Solve first column
            for i in range(1, lpatts):
                DM[p,i,0] += DM[p,i-1,0]
            # Solve the rest
            for i in range(1, lpatts):
                for j in range(1, STS_len):
                    DM[p,i,j] += min([DM[p,i-1,j], w*DM[p,i-1,j-1], w*DM[p,i,j-1]])
    else:
        for p in prange(npatts):
            # First first column
            DM[p,0,0] += w*prev_col[p,0]
            for i in range(1, lpatts):
                DM[p,i,0] += min([
                    DM[p,i-1,0], w*prev_col[p,i-1], w*prev_col[p,i]])
            # Solve first row
            for j in range(1, STS_len):
                DM[p,0,j] += w*DM[p,0,j-1]
            # Solve the rest
            for i in range(1, lpatts):
                for j in range(1, STS_len):
                    DM[p,i,j] += min([
                        DM[p,i-1,j], w*DM[p,i-1,j-1], w*DM[p,i,j-1]])

    # Return the DM
    return DM

# @jit(nopython=True, parallel=True)
# def _minmax(X: np.ndarray):
#     Xmin, Xmax = X[0], X[0]
#     for i in prange(X.shape[0]):
#         if X[i] < Xmin:
#             Xmin = X[i]
#         if X[i] > Xmax:
#             Xmax = X[i]
#     return np.ndarray([Xmin,Xmax])

# @jit(nopython=True, parallel=True)
# def _minmax_clip_scaler(
#         X: np.ndarray, 
#         clip_range: np.ndarray = None,
#         target_range: np.ndarray = np.ndarray([-1,1]),
#         ) -> np.ndarray:
#     if clip_range is None:
#         clip_range = _minmax(X)
#     X_std = (X - clip_range[0]) / (clip_range[1] - clip_range[0])
#     if clip_range is not None:
#         for i in prange(X.shape[0]):
#             if X[i] < clip_range[0]:
#                 X[i] = clip_range[0]
#             if X[i] > clip_range[1]:
#                 X[i] = clip_range[0]
#     return X_std * (target_range[1] - target_range[0]) + target_range[0]

# @jit(nopython=True, parallel=True)
# def compute_GM_optim(
#         STS: np.ndarray, 
#         patts: np.ndarray,
#         clip_range: np.ndarray = None
#         ) -> np.ndarray:

#     STS_len: int = STS.shape[1]
#     npatts: int = patts.shape[0]
#     dpatts: int = patts.shape[1]
#     lpatts: int = patts.shape[2]

#     # scale the STS
#     STS_cos = np.zeros_like(STS)
#     if clip_range is None:
#         for j in prange(dpatts):
#             STS_cos[j] = _minmax_clip_scaler(STS[j])
#     else:
#         for j in prange(dpatts):
#             STS_cos[j] = _minmax_clip_scaler(STS[j], clip_range=clip_range[j])
#     STS_sin = np.sqrt(np.clip(1 - STS_cos ** 2, 0, 1))

#     # scale the patterns
#     patts_cos = np.zeros_like(patts)
#     for i in prange(lpatts):
#         for j in prange(dpatts):
#             patts_cos[i,j,:] = _minmax_clip_scaler(patts[i,j,:])
#     patts_sin = np.sqrt(np.clip(1 - patts_cos ** 2, 0, 1))
    
#     # Compute the Gramian summantion distance matrix
#     GM = np.zeros((npatts, lpatts, STS_len), dtype=np.float32)
#     for p in prange(npatts):
#         for i in prange(STS_len):
#             for j in prange(lpatts):
#                 for k in prange(dpatts):
#                     GM[p,j,i] += STS_cos[k,i]*patts_cos[p,k,j]+STS_sin[k,i]*patts_sin[p,k,j]

#     # Return the full distance matrix
#     return GM

if __name__ == "__main__":


    """ Small test script for optimization. """

    import time
    import matplotlib.pyplot as plt 

    STSl = 400

    STS = np.sin(np.linspace(0, 6*np.pi, STSl))
    lpat = 100
    patterns = np.stack([np.arange(0, lpat), np.zeros(lpat), np.arange(0, lpat)[::-1]])
    # standardize patterns
    patterns = (patterns - np.mean(patterns, axis=1, keepdims=True)) / np.std(patterns, axis=1, keepdims=True)
    print(patterns)
    patterns[1] = 0
    patterns = np.expand_dims(patterns, 1)

    STS = np.expand_dims(STS, 0)

    start = time.perf_counter()
    DM_p1 = compute_DM(STS[:,:STSl//2], patterns, 0.1, metric="squared")
    DM_p2 = compute_DM(STS[:,STSl//2:], patterns, 0.1, metric="squared", prev_col=DM_p1[:,:,-1])
    end = time.perf_counter()
    print("Elapsed (with comp) = {}s".format((end - start)))

    start = time.perf_counter()
    DM_p1 = compute_DM(STS[:,:STSl//2], patterns, 0.1, metric="squared")
    DM_p2 = compute_DM(STS[:,STSl//2:], patterns, 0.1, metric="squared", prev_col=DM_p1[:,:,-1])
    end = time.perf_counter()
    print("Elapsed (no comp) = {}s".format((end - start)))

    # start = time.perf_counter()
    # DM2 = compute_DM_optim(STS, patterns, 0.1)
    # end = time.perf_counter()
    # print("Elapsed (with compilation) = {}s".format((end - start)))

    plt.figure()
    plt.plot(STS[0])
    plt.figure()
    plt.plot(patterns[0,0])
    plt.figure()
    plt.plot(patterns[1,0])
    plt.figure()
    plt.plot(patterns[2,0])
    
    plt.figure()
    plt.imshow(DM_p1[0])
    plt.figure()
    plt.imshow(DM_p2[0])

    # plt.figure()
    # plt.imshow(DM2[0])

    
    plt.show()