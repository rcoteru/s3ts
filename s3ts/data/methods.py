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

def split_by_test_subject(sts, test_subject, n_val_subjects, seed=42):
    if hasattr(sts, "subject_indices"):
        subject_splits = sts.subject_indices
    else:
        subject_splits = list(sts.splits)

    rng = np.random.default_rng(seed)

    val_subject_indices = np.arange(len(subject_splits) - 1)
    val_subjects_selected = list(rng.choice(val_subject_indices, n_val_subjects, replace=False))
    
    if not isinstance(test_subject, list):
        test_subject = [test_subject]

    for s in test_subject:
        if s > len(subject_splits) - 1:
            raise Exception(f"No subject with index {s}")

    return {
        "train": lambda x: return_indices_train(x, subjects=test_subject + val_subjects_selected, subject_splits=subject_splits),
        "val": lambda x: return_indices_test(x, subjects=val_subjects_selected, subject_splits=subject_splits),
        "test": lambda x: return_indices_test(x, subjects=test_subject, subject_splits=subject_splits),
    }

def process_fft(STS, SCS):
    class_changes = [0] + list(np.nonzero(np.diff(SCS))[0])

    top10 = {} # a dict for each class
    classes = np.unique(SCS)
    for c in classes:
        top10[c] = [{} for i in range(STS.shape[0])] # a dict for each channel

    for i in range(len(class_changes)-1):
        current_class = SCS[class_changes[i]+1].item()

        series_part = STS[:, (class_changes[i]+1):(class_changes[i+1]+1)]
        fft_size = 2**int(np.log2(series_part.shape[1]))
        fft_short = np.fft.fft(series_part, axis=-1, n=fft_size)
        fft_freq = np.fft.fftfreq(fft_size) # highest frequencies for signals of sampling rate 50 is 25

        fft_short_sort = np.argsort(np.abs(fft_short), axis=-1) # sort is ascending magnitudes

        fft_freq_sort = np.abs(fft_freq[fft_short_sort][:,-5:]) # get the 10 for each channel frequencies (+-) with highest amplitude

        for c in range(fft_freq_sort.shape[0]):
            for j in range(fft_freq_sort.shape[1]):
                top10[current_class][c][fft_freq_sort[c, j]] = top10[current_class][c].get(fft_freq_sort[c, j], 0) + 1

    return top10

def get_predominant_frequency(fft_process_result):
    classes_list = list(filter(lambda x: x!=100, fft_process_result.keys()))
    num_classes = len(classes_list)

    out = np.zeros((num_classes, len(fft_process_result[0]))) # (n, c) we get a predominant frequency per channel, per class

    for i, c in enumerate(classes_list):
        for j, channel_result in enumerate(fft_process_result[c]):
            sorted_fr = list(filter(lambda x: x[0]>0, sorted(channel_result.items(), key=lambda x:x[1])))
            out[i, j] = sorted_fr[-1][0]
    
    return out