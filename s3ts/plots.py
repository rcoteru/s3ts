import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from s3ts.old_tasks import data_folder, base_main_fname, base_task_fname, image_folder


from pathlib import Path
import logging

log = logging.Logger(__name__)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

plt.rcParams['font.size'] = 14
# plt.rcParams['font.family'] = "Gulasch", "Times", "Times New Roman", "serif"
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = 0.5 * plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = 0.5 * plt.rcParams['font.size']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# plot the medoids
# img_folder = exp_path / "images" / "medoids"
# img_folder.mkdir(parents=True, exist_ok=True)
# for i in range(medoids.shape[0]):
#     plt.figure(figsize=(6,6))
#     plt.title(f"Medoid of class {i}")
#     plt.plot(medoids[i])
#     plt.savefig(img_folder / f"medoid{i}.png")
#     plt.close()

def plot_medoids(
        exp_path: Path,
        show: bool = False
        ) -> None:

    # load experiment data
    base_file = exp_path / base_main_fname
    with np.load(base_file) as data:
        medoids, medoid_ids = data["medoids"], data["medoid_ids"]
        X_train, Y_train = data["X_train"], data["Y_train"]
        X_test,  Y_test  = data["X_test"], data["Y_test"]
        n_labels = data["n_labels"]
    
    # plot all the medoids
    plt.figure(figsize=(6,6), layout="constrained")
    for i in range(medoids.shape[0]):
        sns.lineplot(medoids[i], label=f"Class {i} (ID: {medoid_ids[i]})")
    plt.xlabel("Time")

    # save the images
    img_folder = exp_path / image_folder
    img_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_folder / f"medoids.png")

    if show: # show if needed or close it
        plt.show()    
    plt.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def plot_OESM(
        exp_path: Path,
        task_name: str,
        n_patterns: int,
        sts_range: tuple[int, int],
        show: bool = False
        ) -> None:

    # load experiment data
    base_file = exp_path / base_main_fname
    with np.load(base_file) as data:
        medoids, medoid_ids = data["medoids"], data["medoid_ids"]
        X_train, Y_train = data["X_train"], data["Y_train"]
        X_test,  Y_test  = data["X_test"], data["Y_test"]
        n_labels = data["n_labels"]

    task_file = exp_path / f"{task_name}_{base_task_fname}"
    with np.load(task_file) as data:
        STS_train, STS_test = data["STS_train"], data["STS_test"]
        labels_train, labels_test = data["labels_train"], data["labels_test"]
        OESM_train, OESM_test = data["OESM_train"], data["OESM_test"]
    

    fig = plt.figure(figsize=(14, 6), dpi=100, layout="constrained")
    gs = fig.add_gridspec(nrows=n_patterns+1, ncols=2,
            hspace=0, height_ratios=None,
            wspace=0, width_ratios=[0.1, 0.9])

    vlines = np.where(np.mod(np.arange(sts_range[0], sts_range[1]), X_train.shape[1]) == 0)[0]

    sts_ax = fig.add_subplot(gs[0,1:])
    sts_ax.plot(np.arange(sts_range[0], sts_range[1]), 
                STS_test[sts_range[0]:sts_range[1]],
                color="red")
    sts_ax.set_xlim(sts_range[0], sts_range[1]-1)
    sts_ax.set_xticklabels([])
    sts_ax.set_yticklabels([])
    sts_ax.set_xticks([])
    sts_ax.set_yticks([])
    for x in vlines:
        sts_ax.axvline(x + sts_range[0], color="dimgray")
    sts_ax.grid(True)

    for p in range(n_patterns):

        oesm_ax = fig.add_subplot(gs[p+1,1:])
        img = OESM_test[p,:,sts_range[0]:sts_range[1]]
        oesm_ax.imshow(img, aspect="auto", cmap="Greys")
        oesm_ax.set_yticklabels([])
        oesm_ax.set_xticklabels([])
        oesm_ax.set_xticks([])
        oesm_ax.set_yticks([])
        for x in vlines:
            oesm_ax.axvline(x, color="dimgray")

        patt_ax = fig.add_subplot(gs[p+1,0])
        patt_ax.plot(medoids[p,::-1], 
                    np.arange(len(medoids[0])), 
                    color="red")
        patt_ax.set_yticklabels([])
        patt_ax.set_xticklabels([])
        patt_ax.set_yticks([])
        patt_ax.set_xticks([])
        patt_ax.invert_xaxis()
        patt_ax.grid(True)

        if p == n_patterns-1:
            oesm_ax.set_xlabel("Time")

    # save the image
    img_folder = exp_path / image_folder
    img_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_folder / f"OESM.png", dpi=300)

    if show: # show if needed or close it
        plt.show()    
    plt.close()