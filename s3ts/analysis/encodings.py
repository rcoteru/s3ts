#/usr/bin/env python3
# -*- coding: utf-8 -*-

# package imports
from s3ts.data.acquisition import download_dataset
from s3ts.data.setup import train_test_splits
from s3ts.data.setup import setup_train_dm 

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def encoding_plot(dataset: str,
        folder: Path = Path("figures/encodings"),
        ) -> None:
    
    """ Generates the encoding plots of a given dataset. """

    # dataset settings
    cv_rep          = 0
    exc             = 32
    random_state    = 0
    rho_dfs         = 0.1

    # plot settings
    shift, n_frames = 0, 6
    vmin, vmax = -2.5, 2.5
    print_lines = False
    fontsize = 18
    ts_color = "darkslategray"
    cmaps = ["seismic", "seismic"]
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    # load the dataset
    X, Y, medoids, medoid_idx = download_dataset(dataset=dataset, storage_dir=Path("storage"))
    for j, (train_idx, test_idx) in enumerate(train_test_splits(X, Y, exc=exc, nreps=cv_rep+1, random_state=random_state)):
                if j == cv_rep:
                    break

    # setup the datamodules
    dm_df = setup_train_dm(X=X, Y=Y, patterns=medoids, train_idx=train_idx, test_idx=test_idx,
        test_sts_length=200, train_event_mult=4, train_strat_size=2, train_event_limit=exc,
        batch_size=128, val_size=0.25, rho_dfs=rho_dfs, window_length=10, mode="df", num_workers=4,
        window_time_stride=7, window_patt_stride=1,stride_series=False, random_state=random_state)
    dm_gf = setup_train_dm(X=X, Y=Y, patterns=medoids, train_idx=train_idx, test_idx=test_idx,
        test_sts_length=200, train_event_mult=4, train_strat_size=2, train_event_limit=exc,
        batch_size=128, val_size=0.25, rho_dfs=rho_dfs, window_length=10, mode="gf", num_workers=4,
        window_time_stride=7, window_patt_stride=1,stride_series=False, random_state=random_state)

    # load the required data
    n_patterns = dm_df.n_patterns
    sts_range = (dm_df.l_events*shift, dm_df.l_events*shift + dm_df.l_events*n_frames)
    patts = dm_df.patterns.numpy()
    TS = dm_df.STS_train[sts_range[0]:sts_range[1]]
    DF = dm_df.DM_train[:,:,sts_range[0]:sts_range[1]]
    GF = dm_gf.DM_train[:,:,sts_range[0]:sts_range[1]]

    fig = plt.figure(figsize=(12, 1.5+n_patterns*1.5), dpi=100, layout="constrained")
    gs = fig.add_gridspec(nrows=n_patterns+1, ncols=3,
            hspace=0, height_ratios=None,
            wspace=0, width_ratios=[0.1, 0.45, 0.45])

    vlines = np.where(np.mod(np.arange(sts_range[0], sts_range[1]), dm_df.l_events) == 0)[0]

    corner = fig.add_subplot(gs[0,0])
    corner.set_xticklabels([]), corner.set_yticklabels([])
    corner.set_xticks([]), corner.set_yticks([])
    corner.text(0.95, 0.5, "STS", fontsize=fontsize, rotation=90, ha="center", va="center")
    corner.text(0.5, 0.05, "Patterns", fontsize=fontsize, ha="center", va="center")
    corner.spines['top'].set_visible(False)
    corner.spines['right'].set_visible(False)
    corner.spines['bottom'].set_visible(False)
    corner.spines['left'].set_visible(False)

    sts_ax1 = fig.add_subplot(gs[0,1])
    sts_ax1.plot(np.arange(sts_range[0], sts_range[1]), TS, color=ts_color)
    sts_ax1.set_xlim(sts_range[0], sts_range[1]-1)
    sts_ax1.set_xticklabels([]), sts_ax1.set_yticklabels([])
    sts_ax1.set_xticks([]), sts_ax1.set_yticks([])
    if print_lines:
            [sts_ax1.axvline(x + sts_range[0], color="dimgray", linewidth=0.5) for x in vlines]
    sts_ax1.grid(True)

    sts_ax2 = fig.add_subplot(gs[0,2])
    sts_ax2.plot(np.arange(sts_range[0], sts_range[1]), TS, color=ts_color)
    sts_ax2.set_xlim(sts_range[0], sts_range[1]-1)
    sts_ax2.set_xticklabels([]), sts_ax2.set_yticklabels([])
    sts_ax2.set_xticks([]), sts_ax2.set_yticks([])
    if print_lines:
            [sts_ax2.axvline(x + sts_range[0], color="dimgray", linewidth=0.5) for x in vlines]
    sts_ax1.grid(True)

    for p in range(n_patterns):

            # patt plot
            patt_ax = fig.add_subplot(gs[p+1,0])
            patt_ax.plot(patts[p,::-1], np.arange(len(patts[0])), color=ts_color)
            patt_ax.set_yticklabels([]), patt_ax.set_xticklabels([])
            patt_ax.set_yticks([]), patt_ax.set_xticks([])
            patt_ax.invert_xaxis(), patt_ax.grid(True)

            # df plot
            df_ax = fig.add_subplot(gs[p+1,1])        
            df_im = DF[p,:,sts_range[0]:sts_range[1]]
            df_im = 2*(df_im-df_im.mean())/df_im.std()
            df_ax.imshow(df_im, aspect="auto", cmap=cmaps[0], vmin=vmin, vmax=vmax)
            df_ax.set_yticklabels([]), df_ax.set_xticklabels([])
            df_ax.set_xticks([]), df_ax.set_yticks([])

            # gf plot
            gf_ax = fig.add_subplot(gs[p+1,2])
            gf_im = GF[p,:,sts_range[0]:sts_range[1]]
            gf_im = 2*(gf_im-gf_im.mean())/gf_im.std()
            gf_ax.imshow(gf_im, aspect="auto", cmap=cmaps[1], vmin=vmin, vmax=vmax)
            gf_ax.set_yticklabels([]), gf_ax.set_xticklabels([])
            gf_ax.set_xticks([]), gf_ax.set_yticks([])

            patt_ax.set_ylabel("Class {}".format(p+1), fontsize=fontsize)

            if p == n_patterns-1:
                    df_ax.set_xlabel("DF Representation", fontsize=fontsize)
                    gf_ax.set_xlabel("GF Representation", fontsize=fontsize)

    plt.savefig(folder / f"enc_{dataset}.pdf", bbox_inches="tight")
