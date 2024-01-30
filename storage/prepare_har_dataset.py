import pandas
import numpy as np
import os
import datetime

import re
import wget

import zipfile
import tarfile

DATASETS = {
    "WARD": "https://people.eecs.berkeley.edu/~yang/software/WAR/WARD1.zip",
    "HASC": "http://bit.ly/i0ivEz",
    "UCI-HAR": "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip",
    "WISDM": "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz", # tar.gz
    "USC-HAD": "https://sipi.usc.edu/had/USC-HAD.zip",
    "OPPORTUNITY": "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
    "UMAFall": "https://figshare.com/ndownloader/articles/4214283/versions/7",
    "UDC-HAR": "https://lbd.udc.es/research/real-life-HAR-dataset/data_raw.zip",
    "HARTH": "http://www.archive.ics.uci.edu/static/public/779/harth.zip",
    "UniMiB-SHAR": "https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip?dl=1",
    "REALDISP": "https://archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip",
    "DAPHNET-FOG": "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip",
    "MHEALTH": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    "TempestaTMD": "https://tempesta.cs.unibo.it/projects/us-tm2017/static/dataset/raw_data/raw_data.tar.gz" # tar
}

def download(dataset_name = "all", dataset_dir=None):

    if dataset_name == "all":
        for name in DATASETS.keys():
            download(dataset_name=name, dataset_dir=dataset_dir)
        return None
    
    assert dataset_name in DATASETS.keys()

    if not os.path.exists(f"{dataset_dir}/{dataset_name}"):
        os.mkdir(f"{dataset_dir}/{dataset_name}")

    #check if directory is empty
    if not os.listdir(f"{dataset_dir}/{dataset_name}/"):
        # download zipped dataset
        print(f"Downloading dataset {dataset_name}")
        file = wget.download(DATASETS[dataset_name], out=f"{dataset_dir}/{dataset_name}/")
        print(f"{dataset_name} downloaded to {file}")

def unpack(dataset_name = "all", dataset_dir=None):

    if dataset_name == "all":
        for name in DATASETS.keys():
            unpack(dataset_name=name, dataset_dir=dataset_dir)
        return None
    
    if not os.path.exists(f"{dataset_dir}/{dataset_name}") or not os.listdir(f"{dataset_dir}/{dataset_name}/"):
        return print(f"Dataset is not downloaded to {dataset_dir}/{dataset_name}/")
    
    assert dataset_name in DATASETS.keys()
    assert os.path.exists(f"{dataset_dir}/{dataset_name}")

    files = os.listdir(f"{dataset_dir}/{dataset_name}")
    assert len(files) > 0

    if len(files) > 1:
        return print(f"Dataset {dataset_name} already unpacked")

    if dataset_name in ["WISDM", "TempestaTMD"]:
        unpack_tar(dataset_dir, dataset_name)
    else:
        print(os.path.join(f"{dataset_dir}/{dataset_name}", files[-1]))
        # use zipfile to decompress
        with zipfile.ZipFile(os.path.join(f"{dataset_dir}/{dataset_name}", files[-1]), "r") as zip_ref:
            zip_ref.extractall(f"{dataset_dir}/{dataset_name}")

        # unpack zip files inside zip file
        files_new = os.listdir(f"{dataset_dir}/{dataset_name}")
        for new_file in files_new:
            if new_file[-3:] == "zip" and new_file not in files:
                with zipfile.ZipFile(os.path.join(f"{dataset_dir}/{dataset_name}", new_file), "r") as zip_ref:
                    zip_ref.extractall(f"{dataset_dir}/{dataset_name}")

def unpack_tar(dataset_dir, dataset_name):
    ds_dir = os.path.join(dataset_dir, dataset_name)
    assert os.path.exists(ds_dir)

    files = os.listdir(ds_dir)
    assert len(files) == 1

    f = tarfile.open(os.path.join(ds_dir, files[0]))

    f.extractall(ds_dir)

############################################################################################

def prepare_harth(dataset_dir):

    ds = []

    counts = {}
    event_length = {}

    for dir, _, files in os.walk(os.path.join(dataset_dir, "harth")):
        files.sort()

        for i, file in enumerate(files):
            print(file)
            ds = pandas.read_csv(os.path.join(dir, file))

            # change timestamp to time (from 0)
            if len(ds["timestamp"][0]) == 29: # one of the .csv files has incorrect formatting on milliseconds
                ds["dt"] = list(map(lambda x : (datetime.datetime.strptime(x[:-3], "%Y-%m-%d %H:%M:%S.%f") - \
                                datetime.datetime.strptime(ds["timestamp"][0][:-3], "%Y-%m-%d %H:%M:%S.%f")) / datetime.timedelta(milliseconds=1), ds["timestamp"]) )
            else:
                ds["dt"] = list(map(lambda x : (datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") - \
                            datetime.datetime.strptime(ds["timestamp"][0], "%Y-%m-%d %H:%M:%S.%f")) / datetime.timedelta(milliseconds=1), ds["timestamp"]) )

            # check if the sampling rate is correct, at 50hz
            # all .csv files have some timestamps with jumps of less or more than 15ms
            # for jumps with less than 10ms, we remove observations, for jumps with more than 25ms, we consider a new STS

            remove = []
            j = 0
            for i in range(len(ds) - 1):
                diff = ds["dt"][i + 1] - ds["dt"][j]
                if diff < 15:
                    remove.append(i + 1)
                else:
                    j = i + 1

            print(f"Removed {len(remove)} observations.")
            ds.drop(remove, inplace=True)
            ds = ds.reset_index()

            splits = []
            last = 0
            for i in range(len(ds) - 1):
                if (ds["dt"][i + 1] - ds["dt"][i]) > 500: # I get the same number of splits for 25ms, 50ms or 100 ms
                    splits.append(ds.loc[last:i])
                    last = i + 1
            splits.append(ds.loc[last:len(ds)])

            if not os.path.exists(os.path.join(dataset_dir, f"{file.replace('.csv', '')}")):
                os.mkdir(os.path.join(dataset_dir, f"{file.replace('.csv', '')}"))

            for i, sp in enumerate(splits):
                labels = sp[["label"]].to_numpy()

                with open(os.path.join(dataset_dir, f"{file.replace('.csv', '')}/acc{i}.npy"), "wb") as f:
                    np.save(f, sp[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].to_numpy())
                
                with open(os.path.join(dataset_dir, f"{file.replace('.csv', '')}/label{i}.npy"), "wb") as f:
                    np.save(f, labels)
                
                # update class counts
                lb, c = np.unique(labels, return_counts=True)
                for i, l in enumerate(lb):
                    counts[l] = counts.get(l, 0) + c[i]

                # update event counts and event length
                current_event = 0
                for i in range(1, labels.size - 1):
                    if labels[i] != labels[current_event]:
                        event_length[int(labels[current_event])] = \
                            event_length.get(int(labels[current_event]), []) + [i - current_event]
                        current_event = i
                
                # last event
                event_length[int(labels[current_event])] = \
                            event_length.get(int(labels[current_event]), []) + [labels.size - current_event]
    
    # print statistics
    total = sum(counts.values())
    print(f"Total number of observations: {total}")

    for c in counts.keys():
            print(f"{len(event_length[c])} events in class {c},")
            print(f"\twith size (min) {min(event_length[c])}, (max) {max(event_length[c])}, (mean) {np.mean(event_length[c])}")
            print(f"\t{counts[c]} observations ({(counts[c]/total):.2f})")


def prepare_uci_har(dataset_dir):
    # load data
    files = os.listdir(os.path.join(dataset_dir, "RawData"))
    files.sort()

    data = {}

    total_points = 0
    for file in files:
        if file == "labels.txt":
            data[file] = pandas.read_csv(os.path.join(dataset_dir, "RawData", file), sep=" ", header=None).to_numpy().astype(np.int64)
            continue

        data[file] = pandas.read_csv(os.path.join(dataset_dir, "RawData", file), sep=" ", header=None).to_numpy()

        obs = data[file].shape[0]
        total_points += obs

    print(total_points)

    # split into subjects
    if not os.path.exists(os.path.join(dataset_dir, "processed")):
        os.mkdir(os.path.join(dataset_dir, "processed"))

    user_splits = {}

    for j, file in enumerate([f for f in files if "acc" in f]):
        n_obs = data[file].shape[0]

        re_result = re.match(r"acc_exp(\d+)_user(\d+).txt", file)

        experiment_number = re_result.group(1)
        user_id = re_result.group(2)

        exp_n_int = int(experiment_number)
        user_id_int = int(user_id)

        if not os.path.exists(os.path.join(dataset_dir, "processed", f"subject{user_id}")):
            os.mkdir(os.path.join(dataset_dir, "processed", f"subject{user_id}"))
        
        split_n = user_splits.get(user_id, 0)
        sensor_dir = os.path.join(dataset_dir, "processed", f"subject{user_id}", f"sensor{split_n}.npy")
        label_dir = os.path.join(dataset_dir, "processed", f"subject{user_id}", f"label{split_n}.npy")

        experiment_labels = np.zeros(n_obs)

        for row in data["labels.txt"]:
            exp_id, u_id, label, start, end = row
            if exp_id == exp_n_int and u_id == user_id_int:
                experiment_labels[start:(end+1)] = label

        with open(sensor_dir, "wb") as f:
            np.save(f, np.hstack((data[file], data[file.replace("acc", "gyro")])))
                    
        with open(label_dir, "wb") as f:
            np.save(f, experiment_labels)

        user_splits[user_id] = user_splits.get(user_id, 0) + 1


def prepare_mhealth(dataset_dir):
    files = os.listdir(os.path.join(dataset_dir, "MHEALTHDATASET")) # contains .log files
    # Filter the files
    files = [f for f in files if ".log" in f]
    files.sort()

    assert len(files) == 10 # dataset has 10 subjects

    # Assume no missing data
    # extract features
    columns = [
        "chest_acc_x", "chest_acc_y", "chest_acc_z",
        "ecg_1", "ecg_2",
        "lankle_acc_x", "lankle_acc_y", "lankle_acc_z",
        "lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z",
        "lankle_mag_x", "lankle_mag_y", "lankle_mag_z", 
        "rankle_acc_x", "rankle_acc_y", "rankle_acc_z",
        "rankle_gyro_x", "rankle_gyro_y", "rankle_gyro_z",
        "rankle_mag_x", "rankle_mag_y", "rankle_mag_z",
        "label" 
    ]

    for i, f in enumerate(files):
        ds = pandas.read_csv(os.path.join(dataset_dir, "MHEALTHDATASET", f), header=None, sep="\t")
        ds.columns = columns

        # save acc and other sensor data separately
        with open(os.path.join(dataset_dir, f"chest_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["chest_acc_x", "chest_acc_y", "chest_acc_z"]].to_numpy())
        
        with open(os.path.join(dataset_dir, f"acc_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["lankle_acc_x", "lankle_acc_y", "lankle_acc_z",
                                  "rankle_acc_x", "rankle_acc_y", "rankle_acc_z"]].to_numpy())
        
        with open(os.path.join(dataset_dir, f"mag_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["lankle_mag_x", "lankle_mag_y", "lankle_mag_z",
                                  "rankle_mag_x", "rankle_mag_y", "rankle_mag_z"]].to_numpy())
            
        with open(os.path.join(dataset_dir, f"gyro_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z",
                                  "rankle_gyro_x", "rankle_gyro_y", "rankle_gyro_z"]].to_numpy())

        with open(os.path.join(dataset_dir, f"ecg_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["ecg_1", "ecg_2"]].to_numpy())

        with open(os.path.join(dataset_dir, f"labels_subject{i}.npz"), "wb") as savefile:
            np.save(savefile, ds[["label"]].to_numpy())    

def prepare_wisdm(dataset_dir):
    data_dir = os.path.join(dataset_dir, "WISDM_ar_v1.1")
    assert os.path.exists(data_dir)

    # clean original csv
    with open(os.path.join(data_dir, "WISDM_ar_v1.1_raw.txt"), "r") as f:
        data = f.read()
    lines = list(map(lambda x: x.strip().strip(","), data.split(";")))

    data_new = "\n".join(lines)
    with open(os.path.join(data_dir, "clean.csv"), "w") as f:
        f.write(data_new)
    
    act_label = {'Downstairs':0, 'Jogging':1, 'Sitting':2, 'Standing':3, 'Upstairs':4, 'Walking':5}
    df = pandas.read_csv(os.path.join(data_dir, "clean.csv"), 
        header=None, names=["USER", "ACTIVITY", "TIMESTAMP", "acc_x", "acc_y", "acc_z"])
    
    df = df.dropna()

    for user_id in df["USER"].unique(): # order of appearance in the file, so it does not change
        user = df[df["USER"] == user_id]
        usersorted = user.sort_values("TIMESTAMP")

        acc_data = usersorted[["acc_x", "acc_y", "acc_z"]].to_numpy()
        timestamps = usersorted["TIMESTAMP"].to_numpy() / 50000000
        diff_ts = np.diff(timestamps)

        assert np.sum((diff_ts<1.05)*(diff_ts>0.95))

        labels = list(map(lambda x: act_label[x], usersorted["ACTIVITY"]))

        with open(os.path.join(dataset_dir, f"subject_{user_id}_sensor.npy"), "wb") as f:
            np.save(f, acc_data)
        with open(os.path.join(dataset_dir, f"subject_{user_id}_class.npy"), "wb") as f:
            np.save(f, np.array(labels, dtype=int))


def prepare_realdisp(dataset_dir):
    files = [f for f in os.listdir(dataset_dir) if "subject" in f]
    ideal = list(filter(lambda x: "ideal" in x, files))
    self = list(filter(lambda x: "self" in x, files))
    mutual = list(filter(lambda x: "mutual" in x, files))
    ideal.sort()
    self.sort()
    mutual.sort()

    assert len(ideal) == 17

    if not os.path.exists(os.path.join(dataset_dir, "cleaned")):
        os.mkdir(os.path.join(dataset_dir, "cleaned"))
    
    for file in ideal:
        user = file.replace("_ideal.log", "")
        self = file.replace("ideal", "self")
        mutual_data_files = [f for f in mutual if user+"_" in f]

        extract_and_save_sensor_data(os.path.join(dataset_dir, file), user, "ideal")
        extract_and_save_sensor_data(os.path.join(dataset_dir, self), user, "self")
        for mutual_file in mutual_data_files:
            extract_and_save_sensor_data(os.path.join(dataset_dir, mutual_file), user, mutual_file.replace(".log", "").replace(user + "_", ""))

def extract_and_save_sensor_data(directory, user, mode):
    data = pandas.read_csv(directory, sep="\t", header=None)

    # save accelerometer data by position of the sensor and sensor info
    # i.e. one file per sensor location (RUA or LLA, for example) and one file per sensor (ACC or GYRO, for instance)

    positions = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]
    sensor = ["acc_x", "acc_y", "acc_z", 
              "gyro_x", "gyro_y", "gyro_z", 
              "mag_x", "mag_y", "mag_z", 
              "q1", "q2", "q3", "q4"]

    for i, position in enumerate(positions):
        for j, s in enumerate(["acc", "gyro", "mag", "q"]):
            with open(os.path.join(os.path.dirname(directory), "cleaned", f"{user}_{mode}_{position}_{s}.npy"), "wb") as f:
                if j == 3:
                    numpy_data = data[[i*13+2+3*j, i*13+3+3*j, i*13+4+3*j, i*13+5+3*j]].to_numpy(dtype=np.float32)
                else:
                    numpy_data = data[[i*13+2+3*j, i*13+3+3*j, i*13+4+3*j]].to_numpy(dtype=np.float32)
                np.save(f, numpy_data)

    with open(os.path.join(os.path.dirname(directory), "cleaned", f"{user}_{mode}_labels.npy"), "wb") as f:
        np.save(f, data[[119]].to_numpy(dtype=np.int64))

if __name__ == "__main__":
    download("UCI-HAR", "./datasets")
    download("HARTH", "./datasets")
    # download("MHEALTH", "./datasets")
    download("WISDM", "./datasets")
    #download("REALDISP", "./datasets")

    unpack("UCI-HAR", "./datasets")
    unpack("HARTH", "./datasets")
    # unpack("MHEALTH", "./datasets")
    unpack("WISDM", "./datasets")
    #unpack("REALDISP", "./datasets")

    prepare_uci_har("./datasets/UCI-HAR") # unneccesary
    prepare_harth("./datasets/HARTH")
    # prepare_mhealth("./datasets/MHEALTH")
    prepare_wisdm("./datasets/WISDM")
    #prepare_realdisp("./datasets/REALDISP")

    pass
