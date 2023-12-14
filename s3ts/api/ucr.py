# /usr/bin/env python3
# # -*- coding: utf-8 -*-

import aeon.datasets as aeon
from pathlib import Path
import numpy as np 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def load_ucr_classification(dset: str,
        ) -> tuple[np.ndarray, np.ndarray, dict]:
    
    dset_dir = Path(__file__).parent.resolve()
    dset_dir = dset_dir.parent.parent / "datasets"
    dset_file = dset_dir /f"{dset}.npz"

    if dset_file.exists():
        # load dataset from cache
        print(f"Loading '{dset}' from cache...")
        with np.load(dset_file, allow_pickle=True) as data:
            X, Y, mapping = data["X"], data["Y"], data["mapping"]
    else:
        # download TS dataset from UCR
        print(f"Downloading '{dset}' from UCR...")
        X, Y = aeon.load_classification(name=dset, split=None, return_metadata=False)
        X: np.ndarray = X.astype(np.float32)
        Yn = np.zeros_like(Y, dtype=np.int8)
        mapping = dict()
        for i, y in enumerate(np.unique(Y)):
            mapping[i] = Y
            Yn[Y == y] = i
        Y = Yn

        # # exceptions
        # X, Y = dset_exceptions(dset, X, Y)

        # save the dataset
        np.savez_compressed(dset_file, 
            X=X, Y=Y, mapping=mapping)

    # Return dataset features and labels
    return X, Y, mapping

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# def dset_exceptions(dataset: str, X: np.ndarray, Y: np.ndarray):

#     """ Exceptions for datasets. """

#     if dataset == "Plane":

#         # remove class 4
#         for i in [4]:
#             X = X[Y != i]
#             Y = Y[Y != i]

#         # ensure label consistency
#         Yn = np.zeros_like(Y, dtype=np.int8)
#         for i, y in enumerate(np.unique(Y)):
#             Yn[Y == y] = i
#         Y = Yn

#     elif dataset == "Trace":
        
#         # remove classes 2 and 3
#         X = X[Y != 2]
#         Y = Y[Y != 2]
#         X = X[Y != 3]
#         Y = Y[Y != 3]

#         # ensure label consistency
#         Yn = np.zeros_like(Y, dtype=np.int8)
#         for i, y in enumerate(np.unique(Y)):
#             Yn[Y == y] = i
#         Y = Yn
    
#     elif dataset == "OSULeaf":
        
#         # remove class 5
#         X = X[Y != 5]
#         Y = Y[Y != 5]

#         # ensure label consistency
#         Yn = np.zeros_like(Y, dtype=np.int8)
#         for i, y in enumerate(np.unique(Y)):
#             Yn[Y == y] = i
#         Y = Yn

#     elif dataset == "ECG5000":
        
#         # remove class 4
#         X = X[Y != 4]
#         Y = Y[Y != 4]

#         # ensure label consistency
#         Yn = np.zeros_like(Y, dtype=np.int8)
#         for i, y in enumerate(np.unique(Y)):
#             Yn[Y == y] = i
#         Y = Yn

#     return X, Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
