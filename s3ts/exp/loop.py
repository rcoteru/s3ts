#/usr/bin/env python3
# -*- coding: utf-8 -*-


from s3ts.exp.settings import ExperimentSettings, SlurmSettings

from s3ts.api.nets.methods import create_model_from_DM
from s3ts.api.ucr import load_ucr_classification

from s3ts.api.ts2sts import compute_medoids
from s3ts.api.dms.simulator import DynamicDM

from typing import Optional
from pathlib import Path
import numpy as np
import subprocess
import yaml
import time


def generate_split(X, Y, es: ExperimentSettings) -> dict[str, np.ndarray]:

    """ Generate the train / test split from the data. """

    # Check the shape of the dataset and labels match
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of events in the dataset and labels must be the same.")

    # Check the number of events per class is not larger than the total number of events
    if exc > X.shape[0]:
        raise ValueError("The number of events per class cannot be larger than the total number of events.")
    
    # Check the number of events per class is not larger than the number of events per class
    if exc > np.unique(Y, return_counts=True)[1].min():
        raise ValueError("The number of events per class cannot be larger than the minimum number of events per class.")

    idx = np.arange(X.shape[0])
    rng = np.random.default_rng(random_state)

    for _ in range(nreps):
        
        train_idx = []
        for c in np.unique(Y):
            train_idx.append(rng.choice(idx, size=exc, p=(Y==c).astype(int)/sum(Y==c), replace=False))
        train_idx = np.concatenate(train_idx)
        test_idx = np.setdiff1d(idx, train_idx)

        return train_idx, test_idx

    # if pret:
    #     pass
    
    # else:
    #     for j, (train_idx, test_idx) in enumerate(
    #         train_test_splits(X, Y, exc=exc, nreps=cv_rep+1, random_state=es.seed)):
    #             if j == es.cv_rep:
    #                 break
    #     data_split = {
    #     "train":np.zeros(0),
    #     "val":np.zeros(0),
    #     "test":np.zeros(0)}



def save_results_file(exp_settings: ExperimentSettings,
        
        ) -> None:



    pass


def experiment_loop(exp_settings: ExperimentSettings):
    
    # Print input parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    print(exp_settings)
    es = exp_settings

    # Encoder
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Prepare the data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    start_time = time.perf_counter() # starting time

    # load dataset
    X, Y, mapping = load_ucr_classification(dset=es.dset)

    # compute class medoids to use as patterns
    meds, meds_idx = compute_medoids(X, Y, meds_per_class=1, metric="dtw")
    patts = meds.squeeze(es.num_med)

    # If arch is 'nn', set the window length to the length of the samples
    es.wdw_len = X.shape[2] if es.arch == "nn" else es.wdw_len

    # select the correct image type
    img_type = None if es.dsrc == "ts" else es.dsrc

    # TODO: make data_split
    data_split = generate_split(X, Y, es.cv_rep, pret=es.pret, seed=es.seed)
    if es.pret:
        nsamps = {
            "train": 1,
            "val": 2,
            "test": 3,
        }
    else:
        nsamps = {
            "train": 1,
            "val": 2,
            "test": 3,
        }
    pskip = 1
    
    # generate a dm
    dm = DynamicDM(X=X, Y=Y, patts=patts, img_type=img_type,
        wdw_len=es.wdw_len, wdw_str=es.wdw_str, sts_str=es.str_sts, 
        data_split=data_split, nsamps=nsamps, batch_size=es.batch_size,
        pskip=pskip, seed=es.seed)
    
    dm_time = time.perf_counter() # complete dm time
    
    # directory
    directory = train_dir / "pretrain" / f"{dataset}_{mode}_{arch}"
    directory = train_dir / "finetune" / f"{dataset}_{mode}_{arch}"
    # version = f"exc{exc}_avev{train_exc_limit}_tstrat{train_strat_size}" +\
    #             f"_tmult{train_event_mult}_ststest{test_sts_length}" +\
    #             f"_wlen{window_length}_stride{ss}" +\
    #             f"_wtst{window_time_stride}_wpst{window_patt_stride}" +\
    #             f"_val{val_size}_me{max_epochs}_bs{batch_size}" +\
    #             f"_lr{learning_rate}_rs{random_state}_cv{cv_rep}"
    # version = f"exc{exc}_avev{train_exc_limit}_tstrat{train_strat_size}" +\
    #             f"_tmult{train_event_mult}_ststest{test_sts_length}" +\
    #             f"_wlen{window_length}" +\
    #             f"_val{val_size}_me{max_epochs}_bs{batch_size}" +\
    #             f"_lr{learning_rate}_rs{random_state}_cv{cv_rep}"


    # Do the training
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    model = create_model_from_DM(dm, name=None, 
        dsrc="img", arch="cnn", task="cls")
    
    # Train the model
    data, model = run_model(
        pretrain_mode=pretrain_mode, version=version,
        dataset=dataset, mode=mode, arch=arch, dm=dm, 
        directory=directory, max_epochs=max_epochs,
        learning_rate=learning_rate, encoder_path=encoder_path,
        num_encoder_feats=num_encoder_feats,
        num_decoder_feats=num_decoder_feats,
        random_state=random_state, cv_rep=cv_rep)
    
    train_time = time.perf_counter()
    
    if not pretrain_mode:
        data["exc"] = exc 
        data["train_exc_limit"] = train_exc_limit
        data["train_strat_size"] = train_strat_size
        data["train_event_mult"] = train_event_mult
        data["nevents_test"] = dm.STS_test_events
        data["nevents_train_lim"] = train_event_limit
        data["nevents_train_tot"] = train_event_total

    # Log times
    data["time_dm"] = dm_time - start_time
    data["time_train"] = train_time - dm_time
    data["time_total"] = train_time - start_time

    # Save the results
    save_results(data, res_fname=res_fname, storage_dir=storage_dir)





def run_loop(exp_settings: ExperimentSettings,
        slr_settings: Optional[SlurmSettings] = None):
    """ The magic baby """
    if slr_settings is None:
        return experiment_loop(exp_settings)
    slr_settings.launch_experiment(exp_settings)