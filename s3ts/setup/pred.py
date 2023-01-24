#/usr/bin/python3

from s3ts.frames.tasks.compute import compute_medoids, compute_STS
from s3ts.frames.tasks.download import download_dataset
from s3ts.frames.tasks.oesm import compute_OESM

from s3ts.frames.pred import PredDataModule
from s3ts.frames.base import BaseDataModule

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from pathlib import Path
import numpy as np



def prepare_data_modules(
        
        dataset: str,
    
        batch_size: int,
        window_size: int,
        lab_shifts: list[int],

        rho_dfs: float,
        pret_frac: float,
        test_frac: float,

        nframes_tra: int, 
        nframes_pre: int,
        nframes_test: int,

        seed_sts: int = 0,
        seed_label: int = 0,
        seed_test: int = 0,

        cache_dir: Path = Path("cache")

        ) -> tuple[BaseDataModule, BaseDataModule]:


    print(f"Downloading dataset {dataset}...")
    X, Y, mapping = download_dataset(dataset)

    # set splitting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # división entre train y test
    print(f"Splitting training and test sets (seed: {seed_test})")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
            test_size=test_frac, stratify=Y, random_state=seed_test, shuffle=True)

    # divide en labeled y unlabeled
    print(f"Splitting train and pretrain sets (seed: {seed_label})")
    X_pre, X_tra, Y_pre, Y_tra = train_test_split(X_train, Y_train, 
        test_size=pret_frac, stratify=Y_train, random_state=seed_label, shuffle=True)

    # pattern selection ~~~~~~~~~~~~~~~~~~~~~~~~~

    # selecciona los patrones [n_patterns,  l_patterns]
    print(f"Selecting the DFS patterns from the train data")
    medoids, medoid_ids = compute_medoids(X_tra, Y_tra, distance_type="dtw")

    # STS generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("Generating 'train' STS...")
    STS_tra, labels_tra = compute_STS(X=X_tra,Y=Y_tra, target_nframes=4000, 
        frame_buffer=window_size*3,random_state=seed_sts)

    print("Generating 'pretrain' STS...")
    STS_pre, _, = compute_STS(X=X_pre, Y=Y_pre, target_nframes=4000, 
        frame_buffer=window_size*3,random_state=seed_sts)
    kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile", random_state=random_state)
    kbd.fit(STS_pre.reshape(-1,1))
    labels_pre = kbd.transform(STS_pre.reshape(-1,1)).squeeze().astype(int)
    
    print("Generating 'test' STS...")
    STS_test, labels_test = compute_STS(X=X_test, Y=Y_test, target_nframes=4000, 
        frame_buffer=window_size*3,random_state=seed_sts)

    # DFS generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fracs = f"{test_frac}-{pret_frac}"
    seeds = f"{seed_test}-{seed_label}-{seed_sts}"
    frames = f"{nframes_tra}-{nframes_pre}-{nframes_test}"
    cache_file = cache_dir / f"dfs_{dataset}_{fracs}_{seeds}_{frames}.npz"

    if not Path(cache_file).exists():
        print("Generating 'train' DFS...")
        DFS_tra = compute_OESM(STS_tra, medoids, rho=rho_dfs)   # generate DFS
        print("Generating 'pretrain' DFS...")
        DFS_pre = compute_OESM(STS_pre, medoids, rho=rho_dfs) 
        print("Generating 'test' DFS...")
        DFS_test = compute_OESM(STS_test, medoids, rho=rho_dfs) 
        np.savez_compressed(cache_file)
    else:
        print("Loading DFS from cached file...")
        with np.load(cache_file) as data:
            DFS_tra, DFS_pre, DFS_test = data["DFS_tra"], data["DFS_pre"], data["DFS_test"]

    # create data modules ~~~~~~~~~~~~~~~~~~~~~~~~~~

   
    # UNLABELED DATASET (PRETRAIN) 
    # =================================
    
    print("Creating 'train' dataset...")

    # create data module (train)
    train_dm = PredDataModule(
        STS_tra=STS_tra, DFS_tra=DFS_tra, labels_tra=labels_tra, 
        STS_test=STS_test, DFS_test=DFS_test, labels_test=labels_test, 
        window_size=window_size, batch_size=batch_size)

    print("Creating 'pretrain' dataset...")
    
    l_sample = X.shape[1]
    lab_shifts = np.round(np.array(lab_shifts)*l_sample).astype(int)
    print("Label shifts:", lab_shifts)    

    # create data module (pretrain)
    pretrain_dm = PredDataModule(
        STS=STS_ulab, DFS=DFS_ulab, 
        labels=labels_ulab, 
        window_size=window_size, 
        batch_size=batch_size,
        test_size=test_ratio_ulab,
        lab_shifts=lab_shifts)

   

    return pret_dm, train_dm

   