#/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def timedil_figure(
        df: pd.DataFrame,
        fontsize: int = 18,
        metric: str = "test_acc",
        fname: str = "figures/ablation/timedil.pdf",
        ) -> None:
    
    """ Generates the time dilation figure for the paper. """

    # font settings
    plt.rc('font', family='serif', serif='Times New Roman', size=fontsize)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    # filtering
    data = df[df["pretrain_mode"] == False].copy()
    data = data[data['train_exc_limit'] == 32]
    data = data[data["window_patt_stride"] == 1]
    data = data[data["pretrained"] == False]
    data = data[~data["mode"].isin(["ts"])]
    data = data[data["window_time_stride"].isin([1,3,5,7])]

    data["Method"] =  data["arch"] + "_" + data["mode"]
    data.sort_values(["Method"], inplace=True)
    data["arch"].replace(to_replace=["rnn", "cnn", "res"], value=["RNN", "CNN", "RES"], inplace=True)
    data["mode"].replace(to_replace=["df", "gf"], value=["DF", "GF"], inplace=True)

    # aggregation 
    dfs = []
    for g, gdf in data.groupby(["mode", "arch", "dataset"]):

        bline = gdf.groupby(["mode", "arch", "dataset", "eq_wdw_length"])[metric].mean().iloc[0]
        gdf[metric] = (gdf[metric] - bline)/bline

        mean = gdf.groupby(["mode", "arch", "dataset", "eq_wdw_length"])[metric].mean()
        mean.name = "mean"
        std  = gdf.groupby(["mode", "arch", "dataset", "eq_wdw_length"])[metric].std()
        std.name = "std"
        
        gdat = pd.concat([mean, std], axis=1).reset_index()
        dfs.append(gdat)
    pdata = pd.concat(dfs)

    # figure setup
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,6), gridspec_kw={"wspace": 0.05} )
    ax: list[plt.Axes] 

    cmap = plt.get_cmap("Paired", pdata["dataset"].nunique()*2)
    colors =  {dset: cmap(i*2+1) for i, dset in enumerate(pdata["dataset"].unique())}
    nums =  {dset: i+1 for i, dset in enumerate(pdata["dataset"].unique())}

    for (mode, arch, dset), gdf in pdata.groupby(["mode", "arch", "dataset"]):

        if mode == "DF":
            ax_idx = 0
            color = 0 if arch == "CNN" else 1
        else:
            ax_idx = 1
            color = 2 if arch == "CNN" else 3
        ltype = "solid" if arch == "CNN" else "dashed"
        mtype = "o" if arch == "CNN" else "s"
        color = colors[dset]
        
        iax = ax[ax_idx]
        if ax_idx == 0:
            label = str(nums[dset]) if arch == "CNN" else None
        else:
            label = str(nums[dset]) if arch == "RES" else None
            iax.set_yticklabels([])
        
        iax.set_yticks([-30,-20,-10,0,10,20,30,40,50,60])
        iax.set_xticks([1,3,5,7])
        iax.set_title(mode)
        iax.set_ylim(-40, 70)
        iax.grid(True, axis="y")
        
        iax.axhline(color="black", zorder=-10, lw=3)
        
        iax.errorbar(gdf["eq_wdw_length"]//10, gdf["mean"]*100, yerr=gdf["std"]*100, 
            linestyle=ltype, label=label, c=color, marker=mtype)

    # legends
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="center", bbox_to_anchor=(1.02,0.65), borderpad=0.8, title="CNN", ncols=2, fancybox=False, shadow=True)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="center", bbox_to_anchor=(1.02,0.35), borderpad=0.8, title="RES", ncols=2, fancybox=False, shadow=True)

    # x label
    fig.text(0.512, 0, '$\delta$', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure)
    # y label
    fig.text(0.07, 0.5, r"% Change in Test Accuracy", horizontalalignment='center', verticalalignment='center', 
            transform=fig.transFigure, rotation="vertical")
    
    plt.savefig(fname, bbox_inches='tight')

def pretrain_figure(
        df: pd.DataFrame,
        fontsize: int = 18,
        metric: str = "test_acc",
        fname: str = "figures/ablation/pretrain.pdf",
        ) -> None:
    
    """ Generates the time dilation figure for the paper. """

    # font settings
    plt.rc('font', family='serif', serif='Times New Roman', size=fontsize)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    wlsf  = 10
    wldict = {
        "ArrowHead": 120,
        "CBF": 60,
        "ECG200": 50,
        "ECG5000": 70,
        "GunPoint": 70,
        "SyntheticControl": 30,
        "Trace": 150,
    }

    # cleanup
    data = df[df["pretrain_mode"] == False].copy()
    dfl = []
    for dset, gdf in data.groupby("dataset"):
        wtst =  wldict[dset]//wlsf
        gdf = gdf[(gdf["window_length"] == wlsf) & (gdf["window_time_stride"] == wtst)]
        dfl.append(gdf)
    data = pd.concat(dfl, ignore_index=True)
    #data = data[data["window_time_stride"] == 7]
    data = data[data["window_patt_stride"] == 1]
    data = data[data["arch"].isin(["cnn", "res"])]

    data["Method"] =  data["arch"] + "_" + data["mode"]
    data.sort_values(["Method"], inplace=True)

    data["arch"].replace(to_replace=["cnn", "res"], value=["CNN", "RES"], inplace=True)
    data["mode"].replace(to_replace=["df", "gf"], value=["DF", "GF"], inplace=True)
    data["Arch"] = data["arch"] + data["pretrained"].replace({True: "-", False: ""}) + data["stride_series"].replace({True: "B", False: "A"})
    data["Arch"].replace({"CNNA": "CNN", "RESA": "RES"}, inplace=True)
    # aggregation 
    dfs = []
    for (mode, dataset, arch), gdf in data.groupby(["mode", "dataset", "arch"]):
        mean = gdf.groupby(["mode", "dataset", "Arch", "train_exc_limit"])[metric].mean().reset_index()
        std = gdf.groupby(["mode", "dataset", "Arch", "train_exc_limit"])[metric].std().reset_index()
        
        # baseline
        bline_mean, bline_std = mean[mean["Arch"] == arch][metric].values, std[std["Arch"] == arch][metric].values

        # a pretrain
        a_mean, a_std = mean[mean["Arch"] == f"{arch}-A"][metric].values, std[std["Arch"] == f"{arch}-A"][metric].values

        # b pretrain
        b_mean, b_std = mean[mean["Arch"] == f"{arch}-B"][metric].values, std[std["Arch"] == f"{arch}-B"][metric].values

        gdat = mean[mean["Arch"] == arch][["mode", "dataset", "train_exc_limit"]].copy()
        gdat["arch"] = arch
        gdat["bline_mean"], gdat["bline_std"] = bline_mean, bline_std 
        gdat["a_mean"], gdat["a_std"] = (a_mean - bline_mean)/bline_mean, a_std/a_mean
        gdat["b_mean"], gdat["b_std"] = (b_mean - bline_mean)/bline_mean, b_std/b_mean
        dfs.append(gdat)
    pdata = pd.concat(dfs, ignore_index=True)

    # figure setup
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13,7), gridspec_kw={"wspace": 0.15
                                                                                , "hspace": 0.05})

    dsets = pdata["dataset"].unique()
    cmap = plt.get_cmap("Paired")

    nu_map = {dset: i+1 for i, dset in enumerate(dsets)}
    ax_map = {dset: (idx//ncols, idx%ncols) for idx, dset in enumerate(dsets)}

    for (mode, arch, dset), gdf in pdata.groupby(["mode", "arch", "dataset"]):

        (row, col), dsetn = ax_map[dset], nu_map[dset]
        iax: plt.Axes = ax[row, col]
        
        if mode == "DF":
            color = 0 if arch == "CNN" else 1
        else:
            color = 2 if arch == "CNN" else 3

        if row == 0:
            iax.set_xticklabels([])
        iax.grid(True, axis="y")

        if mode == "DF" and arch == "RES":
            iax.text(0.5, 1, f"{dsetn}", transform=iax.transAxes, fontsize=fontsize-3,
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(edgecolor="lightgray", facecolor='white', alpha=1))
            iax.axhline(color="black", zorder=-10, lw=3)


        # do the plotting    
        x = np.log2(gdf["train_exc_limit"])
        iax.errorbar(x, gdf["a_mean"]*100, yerr=np.abs(gdf["a_mean"]*100*(gdf["a_std"] + gdf["bline_std"]/gdf["bline_mean"])), 
            linestyle="solid", marker="o", c=cmap(color*2+1), label=f"{mode}-{arch}")
        iax.errorbar(x, gdf["b_mean"]*100, yerr=np.abs(gdf["b_mean"]*100*(gdf["b_std"] + gdf["bline_std"]/gdf["bline_mean"])),
            linestyle="dashed", marker="s", c=cmap(color*2), label=f"{mode}-{arch}")
        
            
   # legends
    bpad = 0.9
    handles, labels = ax[0,0].get_legend_handles_labels()
    lidx = np.arange(len(handles))
    fig.legend(handles=handles[::2], labels=labels[::2], loc="center", bbox_to_anchor=(.97,0.65), 
            borderpad=bpad, title="TASK A", ncols=1, fancybox=False, shadow=True, fontsize=fontsize-5)
    fig.legend(handles=handles[1::2], labels=labels[1::2], loc="center", bbox_to_anchor=(.97,0.35), 
            borderpad=bpad, title="TASK B", ncols=1, fancybox=False, shadow=True, fontsize=fontsize-5)

    # labels
    fig.text(0.5, 0.05, r'$\log_2(n_{tr})$', horizontalalignment='center', verticalalignment='center', 
            transform=fig.transFigure, rotation="horizontal")
    fig.text(0.08, 0.5, r"% Change in Test Accuracy", horizontalalignment='center', verticalalignment='center', 
            transform=fig.transFigure, rotation="vertical");

    plt.savefig(fname, bbox_inches="tight")


if __name__ == "__main__":

    df = pd.read_csv("storage/all_results.csv")
    timedil_figure(df)
    pretrain_figure(df)