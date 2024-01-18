import os
import numpy as np
import torch

import re

from s3ts.data.base import STSDataset

# Load datasets predefined

class UCI_HARDataset(STSDataset):

    def __init__(self,
            dataset_dir: str = None,
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
                wsize: window size
                wstride: window stride
        '''

        # load dataset
        files = os.listdir(os.path.join(dataset_dir, "processed"))
        files.sort()
        
        splits = [0]
        self.subject_indices = [0]

        STS = []
        SCS = []

        for f in files:
            # get separated STS
            segments = filter(
                lambda x: "sensor" in x,
                os.listdir(os.path.join(dataset_dir, "processed", f)))

            for s in segments:

                sensor_data = np.load(os.path.join(dataset_dir, "processed", f, s))
                STS.append(sensor_data)
                label_data = np.load(os.path.join(dataset_dir, "processed", f, s.replace("sensor", "label")))
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
        self.indices[self.SCS == 100] = 0 # remove observations with no label
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
