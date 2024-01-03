import numpy as np
from numba import jit, prange

import torch

from typing import Tuple

# Gramian Frames
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@torch.jit.script
def minmax_scaler(X: torch.Tensor, range: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    '''
        Scales the last dimension of X into the given range X has shape (...,d)
        output has the same shape but the last dimension is scaled to the range
    '''
    X_min = X.min(dim=-1, keepdim=True).values
    X_std = (X - X_min)/(X.max(dim=-1, keepdim=True).values - X_min)
    return X_std * (range[1] - range[0]) + range[0]

@torch.jit.script
def gaf_compute(X: torch.Tensor, mode: str = "s", scaling: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    '''
        Computes the Gramian Angular Field of time series X, per sample, per channel
        input has shape (...,N, d, window_size) (any number of leading dimensions)
        output has shape (..., N, d, window_size, window_size)

        First scales the window_size dimension to -1, 1 range and then computes the GAF
        either summation (setting mode to "s" or "summation") or difference, which is the default
        behaviour.
    '''
    X_cos = minmax_scaler(X, scaling)
    X_sin = torch.clamp((1.0 - X_cos.square()).sqrt(), 0, 1)

    if "s" in mode: # summation
        print("hi")
        return X_cos.unsqueeze(-2) * X_cos.unsqueeze(-1) - X_sin.unsqueeze(-2) * X_sin.unsqueeze(-1)
    else: # difference
        return X_sin.unsqueeze(-2) * X_cos.unsqueeze(-1) - X_cos.unsqueeze(-2) * X_sin.unsqueeze(-1)