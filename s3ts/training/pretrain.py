# data
from s3ts.frames.tasks.compute import compute_medoids, compute_STS
from s3ts.frames.tasks.oesm import compute_OESM
from s3ts.frames.base import BaseDataModule

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from pathlib import Path
import numpy as np

def pretrain_data_modules(
        X: np.ndarray,
        Y: np.ndarray,
        ulab_frac: float,
        test_size: float,
        window_size: int,
        batch_size: int,
        rho_dfs: int,
        random_state: int = 0,
        cache_dir: Path = Path("cache")
        ) -> tuple[BaseDataModule, BaseDataModule]:

    if ulab_frac > 0:

        # divide en labeled y unlabeled
        X_lab, X_ulab, Y_lab, Y_ulab = train_test_split(X, Y, 
            test_size=ulab_frac, stratify=Y, random_state=random_state, shuffle=True)

        # LABELED DATASET (TRAIN)
        # =================================

        # labeled train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X_lab, Y_lab, 
            test_size=test_size, stratify=Y_lab, random_state=random_state,  shuffle=True)

        # selecciona los patrones [n_patterns,  l_patterns]
        medoids, medoid_ids = compute_medoids(X_train, Y_train, distance_type="dtw")

        # generate STS
        STS_lab, labels_lab = compute_STS(                     
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test)

        file_lab = "cache/lab.npy"
        if not Path(file_lab).exists(): 
            DFS_lab = compute_OESM(STS_lab, medoids, rho=rho_dfs)   # generate DFS
            np.save(file_lab, DFS_lab)
        else:
            DFS_lab = np.load(file_lab)

        # create data module (train)
        train_dm = BaseDataModule(
            STS=STS_lab, 
            labels=labels_lab, 
            DFS=DFS_lab, 
            window_size=window_size, 
            batch_size=batch_size,
            test_size=test_size)

        # UNLABELED DATASET (PRETRAIN) 
        # =================================

        # unlabeled train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X_ulab, Y_ulab, 
            test_size=test_size, stratify=Y_ulab, random_state=random_state, shuffle=True)

        # generate STS (discarding labels)
        STS_ulab, _ = compute_STS(                     
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test)

        file_ulab = "cache/ulab.npy"
        if not Path(file_ulab).exists(): 
            DFS_ulab = compute_OESM(STS_ulab, medoids, rho=rho_dfs)  # generate DFS
            np.save(file_ulab, DFS_ulab)
        else:
            DFS_ulab = np.load(file_ulab)

        # generate labels
        kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile", random_state=random_state)
        kbd.fit(STS_ulab.reshape(-1,1))
        labels_ulab = kbd.transform(STS_ulab.reshape(-1,1)).squeeze().astype(int)

        # create data module (pretrain)
        pretrain_dm = BaseDataModule(
            STS=STS_ulab, 
            labels=labels_ulab, 
            DFS=DFS_ulab, 
            window_size=window_size, 
            batch_size=batch_size,
            test_size=test_size)

        return pretrain_dm, train_dm

    else:

        # LABELED DATASET (TRAIN)
        # =================================

        # labeled train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
            test_size=test_size, stratify=Y, random_state=random_state,  shuffle=True)

        # selecciona los patrones [n_patterns,  l_patterns]
        medoids, medoid_ids = compute_medoids(X_train, Y_train, distance_type="dtw")

        # generate STS
        STS_lab, labels_lab = compute_STS(                     
            X_train=X_train, Y_train=Y_train,
            X_test=X_test, Y_test=Y_test)

        file_lab = "cache/lab.npy"
        if not Path(file_lab).exists(): 
            DFS_lab = compute_OESM(STS_lab, medoids, rho=rho_dfs)   # generate DFS
            np.save(file_lab, DFS_lab)
        else:
            DFS_lab = np.load(file_lab)

        # create data module (train)
        train_dm = BaseDataModule(
            STS=STS_lab, 
            labels=labels_lab, 
            DFS=DFS_lab, 
            window_size=window_size, 
            batch_size=batch_size,
            test_size=test_size)

        return None, train_dm