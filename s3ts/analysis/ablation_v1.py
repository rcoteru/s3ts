#/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def timedil_figure_vertical(
        df: pd.DataFrame,
        fontsize: int = 18,
        metric: str = "test_acc",
        metric_lab: str = "Test Accuracy",
        fname: str = "figures/ablation/timedilv.pdf",
        ) -> pd.DataFrame:
    
    """ Generates the time dilation figure for the paper. """

    # Filter the data
    data = df[df["pretrain_mode"] == False].copy()
    data = data[data['train_exc_limit'] == 32]
    data = data[data["window_patt_stride"] == 1]
    data = data[data["pretrained"] == False]
    data = data[~data["mode"].isin(["ts"])]
    
    # Convert metric to percentage
    data[metric] = data[metric]*100

    # Generate a plot
    data["Method"] =  data["arch"] + "_" + data["mode"]
    data.sort_values(["Method"], inplace=True)

    data["arch"].replace(to_replace=["rnn", "cnn", "res"], value=["RNN", "CNN", "RES"], inplace=True)
    data["mode"].replace(to_replace=["df", "gf"], value=["DF", "GAF"], inplace=True)
    data["Arch"] = data["arch"]

    sns.set_theme()
    sns.set_style("whitegrid")
    plt.rc('font', family='serif', serif='Times New Roman', size=fontsize)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    g: sns.FacetGrid = sns.relplot(data=data, x="eq_wdw_length", y=metric, hue='Arch', 
        kind="line", col="mode", row="dataset", palette=sns.color_palette("bright"),
        height=2.5, aspect=1.25, legend="auto", markers="True", col_order=["DF", "GAF"], 
        row_order=["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"],
        facet_kws={"despine": False, "margin_titles": True, "legend_out": True,
                   "sharey": False, "sharex": True})
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=fontsize)
    g.set_xlabels("Equivalent Window Length", fontsize=fontsize-2);
    g.set_ylabels(metric_lab, fontsize=fontsize-2);
    g.set(xlim=(5, 75), xticks=[10, 30, 50, 70])
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.legend.set_title("")

    dsets = ["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"]
    ybounds = {
        "ArrowHead": ([35, 60], [40, 45, 50, 55]),
        "CBF": ([35, 75], [40, 50, 60, 70]),
        "ECG200": ([45, 70], [50, 55, 60, 65]),
        "GunPoint": ([45, 70], [50, 55, 60, 65]),
        "SyntheticControl": ([25, 65], [30, 40, 50, 60]),
        "Trace": ([45, 70], [50, 55, 60, 65])}

    for i, row in enumerate(g.axes):
        for j, ax in enumerate(row):
            ax.set_ylim(ybounds[dsets[i]][0])
            ax.set_yticks(ybounds[dsets[i]][1])

    for (row_val, col_val), ax in g.axes_dict.items():
        if col_val in ["GAF"]:
            ax.set_yticklabels([])
        if row_val not in ["CBF", "SyntheticControl"]:
            ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_color("black")
        if row_val == "Trace":
            ax.set_xlabel("")
            if col_val == "DF":
                ax.text(1, -0.32, "Context Size", ha="center", va="center", transform=ax.transAxes, fontsize=fontsize-2)

    g.savefig(fname, bbox_inches='tight')


def timedil_figure_horizontal(
        df: pd.DataFrame,
        fontsize: int = 18,
        metric: str = "test_acc",
        metric_lab: str = "Test Accuracy",
        fname: str = "figures/ablation/timedilh.pdf",
        ) -> pd.DataFrame:
    
    """ Generates the time dilation figure for the paper. """

    # Filter the data
    data = df[df["pretrain_mode"] == False].copy()
    data = data[data['train_exc_limit'] == 32]
    data = data[data["window_patt_stride"] == 1]
    data = data[data["pretrained"] == False]
    data = data[~data["mode"].isin(["ts"])]
    
    # Convert metric to percentage
    data[metric] = data[metric]*100

    # Generate a plot
    data["Method"] =  data["arch"] + "_" + data["mode"]
    data.sort_values(["Method"], inplace=True)

    data["arch"].replace(to_replace=["rnn", "cnn", "res"], value=["RNN", "CNN", "RES"], inplace=True)
    data["mode"].replace(to_replace=["df", "gf"], value=["DF", "GAF"], inplace=True)
    data["Arch"] = data["arch"]

    sns.set_theme()
    sns.set_style("whitegrid")
    plt.rc('font', family='serif', serif='Times New Roman', size=fontsize)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    g: sns.FacetGrid = sns.relplot(data=data, x="eq_wdw_length", y=metric, hue='Arch', 
        kind="line", col="dataset", row="mode", palette=sns.color_palette("bright"),
        height=2.5, aspect=1, legend="auto", markers="True", row_order=["DF", "GAF"], 
        col_order=["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"],
        facet_kws={"despine": False, "margin_titles": True, "legend_out": True,
                   "sharey": False, "sharex": True})
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=fontsize)
    g.set_xlabels("Equivalent Window Length", fontsize=fontsize-2);
    g.set_ylabels(metric_lab, fontsize=fontsize-2);
    g.set(xlim=(5, 75), xticks=[10, 30, 50, 70])
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.legend.set_title("")

    dsets = ["ArrowHead", "CBF", "ECG200", "GunPoint", "SyntheticControl", "Trace"]
    ybound = ([35, 75], [40, 50, 60, 70])

    for i, row in enumerate(g.axes):
        for j, ax in enumerate(row):
            ax.set_ylim(ybound[0])
            ax.set_yticks(ybound[1])

    for (row_val, col_val), ax in g.axes_dict.items():
        if col_val not in ["ArrowHead"]:
            ax.set_yticklabels([])
        if col_val not in ["CBF", "SyntheticControl"]:
            ax.set_ylabel("")
            ax.set_xlabel("")
        for sp in ax.spines.values():
            sp.set_color("black")
        if row_val == "Trace":
            ax.set_xlabel("")
            if col_val == "DF":
                ax.text(1, -0.32, "Context Size", ha="center", va="center", transform=ax.transAxes, fontsize=fontsize-2)

    g.savefig(fname, bbox_inches='tight')