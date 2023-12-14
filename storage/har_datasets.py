import os
import numpy as np
import torch

import re

from torch.utils.data import Dataset

from s3ts.api.ts2sts import compute_medoids
from tslearn.clustering import TimeSeriesKMeans

class STSDataset(Dataset):

    def __init__(self,
            wsize: int = 10,
            wstride: int = 1,
            ) -> None:
        super().__init__()

        '''
            Base class for STS dataset

            Inputs:
                wsize: window size
                wstride: window stride
        '''

        self.wsize = wsize
        self.wstride = wstride

        self.splits = None

        self.STS = None
        self.SCS = None

        self.indices = None

    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:

        first = self.indices[index]-self.wsize*self.wstride
        last = self.indices[index]

        return self.STS[:, first:last:self.wstride], self.SCS[first:last:self.wstride]
    
    def sliceFromArrayOfIndices(self, indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(indexes.shape) == 1 # only accept 1-dimensional arrays

        return_sts = np.empty((indexes.shape[0], self.STS.shape[0], self.wsize))
        return_scs = np.empty((indexes.shape[0], self.wsize))

        for i, id in enumerate(indexes):
            ts, c = self[id]
            return_scs[i] = c
            return_sts[i] = ts

        return return_sts, return_scs
    
    def getSameClassWindowIndex(self):

        id = []
        cl = []
        for i, ix in enumerate(self.indices):
            if np.unique(self.SCS[(ix-self.wsize*self.wstride):ix]).shape[0] == 1:
                id.append(i)
                cl.append(self.SCS[ix])
        
        return np.array(id), np.array(cl)
    
    def normalizeSTS(self, mode):
        self.mean = np.expand_dims(self.STS.mean(1), 1)
        self.percentile1 = np.expand_dims(np.percentile(self.STS, 1, axis=1), 1)
        self.percentile99 = np.expand_dims(np.percentile(self.STS, 99, axis=1), 1)

        self.STS = (self.STS - self.mean) / (self.percentile99 - self.percentile1)

# Methods to obtain patterns

def sts_medoids(dataset: STSDataset, n = 100, pattern_size = -1, random_seed: int = 45):
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
    meds, meds_id = compute_medoids(selected_w, np.concatenate(selected_c, axis=0))

    return meds[:,0,:,:]

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

class StreamingTimeSeries(STSDataset):

    def __init__(self,
            STS: np.ndarray,
            SCS: np.ndarray,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        self.STS = STS
        self.SCS = SCS

        self.splits = np.array([0, SCS.shape[0]])

        # process ds
        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")


class StreamingTimeSeriesCopy(Dataset):

    def __init__(self,
            stsds: StreamingTimeSeries, indices: np.ndarray
            ) -> None:
        super().__init__()

        self.stsds = stsds
        self.indices = indices
        
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:

        ts, c = self.stsds[self.indices[index]]
        return {"series": ts, "label": c[-1]}
    
    def __del__(self):
        del self.stsds

def reduce_imbalance(indices, labels, seed = 42):
    rng = torch.Generator()
    rng.manual_seed(seed)

    cl, counts = torch.unique(labels, return_counts=True)
    mean = counts.float().mean()

    mask = torch.ones_like(labels, dtype=bool)
    for id in torch.argwhere(counts > mean):
        mask[labels == cl[id]] = torch.rand(counts[id], generator=rng) < counts[torch.arange(counts.shape[0]) != id].float().mean()/counts[id]
    
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

# Load datasets predefined

class UCI_HARDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            split: str = "train",
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            UCI-HAR dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                split: "train" or "test"
                wsize: window size
                wstride: window stride
        '''

        assert split in ("both", "train", "test")
        if split == "both":
            split = ["train", "test"]
        else:
            split = [split]
        
        splits = [0]

        STS = []
        SCS = []

        for s in split:

            # load dataset
            files = list(filter(
                lambda x: "sensor.npy" in x,
                os.listdir(os.path.join(dataset_dir, "UCI HAR Dataset", s)))
            )
            files.sort()


            for f in files:
                sensor_data = np.load(os.path.join(dataset_dir, "UCI HAR Dataset", s, f))
                STS.append(sensor_data)
                SCS.append(np.load(os.path.join(dataset_dir, "UCI HAR Dataset", s, f.replace("sensor", "class"))))

                splits.append(splits[-1] + sensor_data.shape[0])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS).T
        self.SCS = np.concatenate(SCS).astype(np.int32)

        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")

class HARTHDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            HARTH dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                wsize: window size
                wstride: window stride
        '''

        # load dataset
        files = list(filter(
            lambda x: ".csv" in x,
            os.listdir(os.path.join(dataset_dir, "harth")))
        )
        files.sort()
        
        splits = [0]

        self.subject_indices = [0]

        STS = []
        SCS = []
        for f in files:
            # get separated STS
            segments = filter(
                lambda x: "acc" in x,
                os.listdir(os.path.join(dataset_dir, f[:-4])))

            for s in segments:

                sensor_data = np.load(os.path.join(dataset_dir, f[:-4], s))
                STS.append(sensor_data)
                label_data = np.load(os.path.join(dataset_dir, f[:-4], s.replace("acc", "label")))
                SCS.append(label_data)

                splits.append(splits[-1] + sensor_data.shape[0])

            self.subject_indices.append(splits[-1])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS).T
        self.SCS = np.squeeze(np.concatenate(SCS).astype(np.int32))

        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")

class MHEALTHDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            sensor: str = "",
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            MHEALTH dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                sensor: what sensor data to load, if empty string, all sensors are used
                wsize: window size
                wstride: window stride
        '''
        
        if isinstance(sensor, list):
            for s in sensor:
                assert s in ("acc", "ecg", "gyro", "mag", "chest")
        elif sensor == "":
            sensor = ["acc", "ecg", "gyro", "mag", "chest"]

        # load dataset
        files = list(filter(
            lambda x: "labels" in x,
            os.listdir(os.path.join(dataset_dir))))
        files.sort()

        splits = [0]

        STS = []
        SCS = []
        for f in files:

            subject = f.replace("labels_", "")
            # get separated STS
            data = []
            for s in sensor:
                data += list(filter(
                    lambda x: (s in x) and (not "labels" in x) and (subject in x),
                    os.listdir(dataset_dir)))
            
            data = sorted(data)

            sensor_data = []
            for s in data:
                sensor_data.append(np.load(os.path.join(dataset_dir, s)))

            sensor_data = np.hstack(sensor_data) # concatenate columns
            STS.append(sensor_data)

            SCS.append(np.load(os.path.join(dataset_dir, f)))
            
            splits.append(splits[-1] + sensor_data.shape[0])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS).T
        self.SCS = np.squeeze(np.concatenate(SCS).astype(np.int32))

        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")

class WISDMDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            WISDM dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                wsize: window size
                wstride: window stride
        '''

        # load dataset
        files = list(filter(
            lambda x: "sensor.npy" in x,
            os.listdir(os.path.join(dataset_dir)))
        )
        files.sort()
        
        splits = [0]

        STS = []
        SCS = []
        for f in files:
            sensor_data = np.load(os.path.join(dataset_dir, f))
            STS.append(sensor_data)
            SCS.append(np.load(os.path.join(dataset_dir, f.replace("sensor", "class"))))

            splits.append(splits[-1] + sensor_data.shape[0])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS).T
        self.SCS = np.concatenate(SCS).astype(np.int32)

        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")


class REALDISPDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
            sensor_position: list[str] = None,
            sensor: list[str] = None,
            mode: list[str] = None,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        '''
            REALDISP dataset handler

            Inputs:
                dataset_dir: Directory of the prepare_har_dataset.py
                    processed dataset.
                wsize: window size
                wstride: window stride
        '''

        if mode is None:
            mode = ["ideal", "self", "mutual"]
        for m in mode:
            assert m in ("ideal", "self", "mutual")

        if sensor is None:
            sensor = ["acc", "gyro", "mag", "q"]
        for s in sensor:
            assert s in ("acc", "gyro", "mag", "q")
            
        if sensor_position is None:
            sensor_position = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]
        for p in sensor_position:
            assert p in ("RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC")

        # extract files
        files = list(filter(
            lambda x: "labels.npy" in x,
            os.listdir(os.path.join(dataset_dir, "cleaned")))
        )
        files_to_load = []

        # extract files corresponding to selected modes
        for m in mode:
            files_to_load += list(filter(lambda x: m in x, files))

        # extract subjects
        subjects = list(set([re.findall("subject[0-9]*", file)[0] for file in files_to_load]))
        subjects.sort()

        splits = [0]

        STS = []
        SCS = []

        self.subject_indices = [0]

        for subject in subjects:
            # files_to_load
            subject_files = [f for f in files_to_load if subject+"_" in f]

            for subject_file in subject_files:
                sensor_data = []
                for position in sensor_position:
                    for s in sensor:
                        sensor_data.append(np.load(os.path.join(dataset_dir, "cleaned", subject_file.replace("labels", f"{position}_{s}"))))
                
                sensor_data = np.hstack(sensor_data)
                STS.append(sensor_data)
                SCS.append(np.load(os.path.join(dataset_dir, "cleaned", subject_file)))

                splits.append(splits[-1] + sensor_data.shape[0])

            self.subject_indices.append(splits[-1])

        self.splits = np.array(splits)

        self.STS = np.concatenate(STS).T
        self.SCS = np.concatenate(SCS).astype(np.int32).reshape(-1)

        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")

if __name__ == "__main__":
    ds = HARTHDataset("./datasets/HARTH/", wsize=64, wstride=1)