#/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

def load_folder(folder: Path) -> pd.DataFrame:

    """ Load all csv files in the results folder into a single dataframe. """

    # load all csv files in the folder using pandas
    df = pd.concat([pd.read_csv(f) for f in folder.glob("*.csv")])

    # fix missing values due to the way the data was saved
    df["stride_series"].replace(np.NaN, False, inplace=True)
    df['train_exc_limit'].replace(np.NaN, 0, inplace=True)
    df["pretrained"].replace(np.NaN, False, inplace=True)
    df["pretrain_mode"].replace(np.NaN, False, inplace=True)
    df["window_time_stride"].replace(np.NaN, 1, inplace=True)
    df["window_patt_stride"].replace(np.NaN, 1, inplace=True)
    df["cv_rep"].replace(np.NaN, False, inplace=True)
    df["eq_wdw_length"] = df["window_length"]*df["window_time_stride"]
    df.sort_values(['mode', 'arch', 'dataset', 'pretrain_mode', 'window_length', "stride_series",
                    'window_time_stride', 'window_patt_stride', 'train_exc_limit', 'pretrained', "cv_rep"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

def duplicate_check(df: pd.DataFrame) -> pd.DataFrame:

    # testing purposes
    # xd = df.groupby(['mode', 'arch', 'dataset', 'pretrain_mode', 'window_length', "stride_series",
    #                 'window_time_stride', 'window_patt_stride', 'train_exc_limit', 'pretrained', "cv_rep"])
    # for _, gdf in xd:
    #     if len(gdf) > 1:
    #         print(gdf[['mode', 'arch', 'dataset', 'pretrain_mode', 'window_length', "stride_series",
    #                 'window_time_stride', 'window_patt_stride', 'train_exc_limit', 'pretrained', "cv_rep"]])

    # check for duplicate entries
    file_entries = len(df)
    df = df.groupby(['mode', 'arch', 'dataset', 'pretrain_mode', 'window_length', "stride_series",
                    'window_time_stride', 'window_patt_stride', 'train_exc_limit', 'pretrained', "cv_rep"]).first().reset_index()
    unique_entries = len(df)
    print(f"{file_entries - unique_entries} duplicate entries removed")
    print(f"{len(df)} total entries")
    return df

def results_table_old(df: pd.DataFrame, 
        metric: str = "test_acc",
        ) -> pd.DataFrame:

    """ Generates the results table for the paper. """

    data = df[df["pretrain_mode"] == False].copy()
    data = data[
        (data["arch"] == "nn") |
        ((data["mode"] == "ts") & (data["window_length"] == 70)) |
        ((data["train_exc_limit"] == 32) & (data["window_length"] == 10) & (data["window_time_stride"] == 7) & (data["window_patt_stride"] == 1))
    ]

    def method(row):
        string = row["mode"].upper() + "-" +  row["arch"].upper() 
        if row["pretrained"]:
            if row["stride_series"]:
                string += "-B"
            else:
                string += "-A"
        return string

    data["method"] = data[["arch", "mode", "pretrained", "stride_series"]].apply(method, axis=1)

    tab1 = (data.groupby(["method", "dataset"])[[metric]].mean()*100).reset_index()
    tab2 = (data.groupby(["method", "dataset"])[[metric]].std()*100).reset_index()
    tab1["var"], tab2["var"] = "mean", "std"

    table = pd.concat([tab1, tab2])
    table = table.groupby(["method", "var", "dataset"])[metric].mean().unstack().unstack().round(0)
    table["avg_rank"] = tab1.groupby(["method", "dataset"])[metric].mean().unstack().rank(ascending=False).mean(axis=1).round(1)

    return table 

def results_table_new(df: pd.DataFrame, 
        metric: str = "test_acc",
        ) -> pd.DataFrame:

    """ Generates the results table for the paper. """

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

    data = df[df["pretrain_mode"] == False].copy()
    dfl = []
    for dset, gdf in data.groupby("dataset"):
        wtst =  wldict[dset]//wlsf
        gdf = gdf[
        (gdf["arch"] == "nn") |
        ((gdf["mode"] == "ts") & (gdf["window_length"] == wldict[dset])) |
        ((gdf["train_exc_limit"] == 32) & (gdf["window_length"] == wlsf) & (gdf["window_time_stride"] == wtst) & (gdf["window_patt_stride"] == 1))
        ]
        dfl.append(gdf)
    data = pd.concat(dfl, ignore_index=True)
    
    def method(row):
        string = row["mode"].upper() + "-" +  row["arch"].upper() 
        if row["pretrained"]:
            if row["stride_series"]:
                string += "-B"
            else:
                string += "-A"
        return string

    data["method"] = data[["arch", "mode", "pretrained", "stride_series"]].apply(method, axis=1)

    tab1 = (data.groupby(["method", "dataset"])[[metric]].mean()*100).reset_index()
    tab2 = (data.groupby(["method", "dataset"])[[metric]].std()*100).reset_index()
    tab1["var"], tab2["var"] = "mean", "std"

    table = pd.concat([tab1, tab2])
    table = table.groupby(["method", "var", "dataset"])[metric].mean().unstack().unstack().round(0)
    table["avg_rank"] = tab1.groupby(["method", "dataset"])[metric].mean().unstack().rank(ascending=False).mean(axis=1).round(1)

    return table

def results_latex(df: pd.DataFrame,
        metric: str = "test_acc") -> str:

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

    # process the data
    data = df[df["pretrain_mode"] == False].copy()
    dfl = []
    for dset, gdf in data.groupby("dataset"):
        wtst =  wldict[dset]//wlsf
        gdf = gdf[
        (gdf["arch"] == "nn") |
        ((gdf["mode"] == "ts") & (gdf["window_length"] == wldict[dset])) |
        ((gdf["train_exc_limit"] == 32) & (gdf["window_length"] == wlsf) & (gdf["window_time_stride"] == wtst) & (gdf["window_patt_stride"] == 1))
        ]
        dfl.append(gdf)
    data = pd.concat(dfl, ignore_index=True)

    # create the aggregation column
    def method(row):
        if row["mode"] != "ts":
            string = row["mode"].upper() + "-" +  row["arch"].upper() 
            if row["pretrained"]:
                if row["stride_series"]:
                    string += "-B"
                else:
                    string += "-A"
            return string
        else:
            tsd = {"rnn": "LSTM", "nn": "NN-DTW", "cnn": "CNN", "res": "RES"}
            string = row["mode"].upper() + "-" + tsd[row["arch"]]
            return string
    data["method"] = data[["arch", "mode", "pretrained", "stride_series"]].apply(method, axis=1)

    # aggregate the means
    tab1 = (data.groupby(["method", "dataset"])[[metric]].mean()*100).reset_index()
    tab2 = (data.groupby(["method", "dataset"])[[metric]].std()*100).reset_index()
    tab3 = tab1.groupby(["method", "dataset"])[metric].mean().unstack().rank(ascending=False).mean(axis=1).round(1)

    dsets = data["dataset"].unique()
    nums = {dset: i+1 for i, dset in enumerate(dsets)}

    ##  generate the latex code
    pm = '{\pm '

    # header
    tab = "\\begin{tabular}{" + "c"*(3 + len(dsets)) + "} \n"
    tab += "&  &  & \multicolumn{" + str(len(dsets))+ "}{c}{\\textbf{DATASET}} \\\\ \n"
    tab += "\\textbf{DTYPE} & \\textbf{ARCH} & \\textbf{RANK}"
    for dset in dsets:
        tab += f" & \\textbf{'{' + str(nums[dset]) + '}'}"
    tab +=  "\\\\ \hline \\\\ \n"

    # ts methods
    tab += "\\multirow{4}{*}{TS}"
    methods = [m for m in data["method"].unique() if "TS" in m]
    methods = [m for _, m in sorted(zip([1,3,0,2], methods), key=lambda pair: pair[0], reverse=True)]
    for m in methods:
        tab +=  f" & {m[3:]} & {tab3[m]}"
        for dset in dsets:
            if len(tab1[(tab1["method"] == m) & (tab1["dataset"] == dset)]) > 0:
                mean = int(tab1[(tab1["method"] == m) & (tab1["dataset"] == dset)].iloc[0][metric].round(0))
                std = int(tab2[(tab2["method"] == m) & (tab2["dataset"] == dset)].iloc[0][metric].round(0))
            else:
                mean, std = 0, 0
            tab += f" & ${mean}_{pm + str(std) + '}'}$"
        tab += " \\\\ \n"
    tab +=" \\\\ \hline \\\\ \n"
    

    # df methods
    for mode in ["DF", "GF"]:
        tab += "\\multirow{6}{*}{" + mode +  "}"
        methods = [m for m in data["method"].unique() if mode in m]
        for m in methods:
            tab +=  f" & {m[3:]} & {tab3[m]}"
            for dset in dsets:
                if len(tab1[(tab1["method"] == m) & (tab1["dataset"] == dset)]) > 0:
                    mean = int(tab1[(tab1["method"] == m) & (tab1["dataset"] == dset)].iloc[0][metric].round(0))
                    std = int(tab2[(tab2["method"] == m) & (tab2["dataset"] == dset)].iloc[0][metric].round(0))
                else:
                    mean, std = 0, 0
                tab += f" & ${mean}_{pm + str(std) + '}'}$"
            tab += " \\\\ \n"
        
        if mode == "DF":
            tab +=" \\\\ \hline \\\\ \n"

    tab += "\\end{tabular}"
    
    return tab


if __name__ == "__main__":

    df1 = load_folder(Path("storage/synced"))
    df2 = load_folder(Path("storage/new"))

    duplicate_check(df1)
    duplicate_check(df2)

    df = pd.concat([df1, df2])
    duplicate_check(df)

