import numpy as np
import torch

from s3ts.data.base import STSDataset
from s3ts.api.ts2sts import compute_medoids

from tslearn.clustering import TimeSeriesKMeans

# Methods to obtain patterns

def sts_medoids(dataset: STSDataset, n = 100, pattern_size = -1, meds_per_class = 1, random_seed: int = 45):
    np.random.seed(random_seed)

    window_id, window_lb = dataset.getSameClassWindowIndex()

    selected_w = []
    selected_c = []

    for i, c in enumerate(np.unique(window_lb)):
        # get the random windows for the class c

        rw = np.random.choice(window_id[window_lb == c].reshape(-1), n)

        ts, cs = dataset.sliceFromArrayOfIndices(rw)

        selected_w.append(ts)
        selected_c.append(np.full(n, c, np.int32))

    selected_w = np.concatenate(selected_w) # (n, dims, len)
    if pattern_size>0:
        selected_w = selected_w[:,:,-pattern_size:]
    meds, meds_id = compute_medoids(selected_w, np.concatenate(selected_c, axis=0), meds_per_class=meds_per_class)

    return meds.reshape((meds.shape[0]*meds.shape[1], meds.shape[2], meds.shape[3]))


def sts_barycenter(dataset: STSDataset, n: int = 100, random_seed: int = 45):
    np.random.seed(random_seed)
    
    window_id, window_lb = dataset.getSameClassWindowIndex()
    selected = np.empty((np.unique(window_lb).shape[0], dataset.STS.shape[0], dataset.wsize))

    for i, c in enumerate(np.unique(window_lb)):
        # get the random windows for the class c

        rw = np.random.choice(window_id[window_lb == c].reshape(-1), n)

        ts, cs = dataset.sliceFromArrayOfIndices(rw)

        km = TimeSeriesKMeans(n_clusters=1, verbose=True, random_state=1, metric="dtw", n_jobs=-1)
        km.fit(np.transpose(ts, (0, 2, 1)))

        selected[i] = km.cluster_centers_[0].T

    return selected


def reduce_imbalance(indices, labels, seed = 42):
    rng = torch.Generator()
    rng.manual_seed(seed)

    cl, counts = torch.unique(labels, return_counts=True)
    median = counts.float().median()

    mask = torch.ones_like(labels, dtype=bool)
    for id in torch.argwhere(counts > median):
        mask[labels == cl[id]] = torch.rand(counts[id], generator=rng) < median/counts[id]
    
    return indices[mask]


def return_indices_train(x, subjects, subject_splits):
    out = np.ones_like(x, dtype=bool)
    for s in subjects:
        out = out & ((x<subject_splits[s]) | (x>subject_splits[s+1]))
    return out

def return_indices_test(x, subjects, subject_splits):
    out = np.zeros_like(x, dtype=bool)
    for s in subjects:
        out = out | ((x>subject_splits[s]) & (x<subject_splits[s+1]))
    return out

# series splitting functions

def split_by_test_subject(sts, subject):
    if hasattr(sts, "subject_indices"):
        subject_splits = sts.subject_indices
    else:
        subject_splits = list(sts.splits)
    
    if isinstance(subject, list):

        for s in subject:
            if s > len(subject_splits) - 1:
                raise Exception(f"No subject with index {s}")

        return {
            "train": lambda x: return_indices_train(x, subjects=subject, subject_splits=subject_splits),
            "val": lambda x: return_indices_test(x, subjects=subject, subject_splits=subject_splits),
            "test": lambda x: return_indices_test(x, subjects=subject, subject_splits=subject_splits),
        }

    else:
        if subject > len(subject_splits) - 1:
            raise Exception(f"No subject with index {subject}")
        
        return {
            "train": lambda x: (x<subject_splits[subject]) | (x>subject_splits[subject+1]),
            "val": lambda x: (x>subject_splits[subject]) & (x<subject_splits[subject+1]),
            "test": lambda x: (x>subject_splits[subject]) & (x<subject_splits[subject+1])
        }
