import os
from time import time

str_time = lambda b: f"{int(b//3600):02d}:{int((b%3600)//60):02d}:{int((b%3600)%60):02d}.{int(round(b%1, 3)*1000):03d}"

from argparse import ArgumentParser
import multiprocessing
import pandas as pd

# dataset imports
from s3ts.data.har_datasets import *
from s3ts.data.dfdataset import LDFDataset, DFDataset
from s3ts.data.stsdataset import LSTSDataset
from s3ts.data.label_mappings import *
from s3ts.data.methods import *

from s3ts.api.nets.methods import create_model_from_DM

from torchvision.transforms import Normalize

def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str,
        help="Dataset name for training")
    parser.add_argument("--dataset_dir", default="./datasets", type=str, 
        help="Directory of the dataset")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=multiprocessing.cpu_count()//2, type=int)
    parser.add_argument("--window_size", default=32, type=int, 
        help="Window size of the dissimilarity frames fed to the classifier")
    parser.add_argument("--window_stride", default=1, type=int, 
        help="Stride used when extracting windows")
    parser.add_argument("--normalize", action="store_true", 
        help="Wether to normalize the dissimilarity frames and STS")
    parser.add_argument("--pattern_size", default=32, type=int, 
        help="Size of the pattern for computation of dissimilarity frames (not used)")
    parser.add_argument("--compute_n", default=500, type=int, 
        help="Number of samples extracted from the STS or Dissimilarity frames to compute medoids and/or means for normalization")
    parser.add_argument("--subjects_for_test", nargs="+", type=int, 
        help="Subjects reserved for testing and validation")
    parser.add_argument("--encoder_architecture", default="cnn", type=str, 
        help="Architecture used for the encoder")
    parser.add_argument("--decoder_architecture", default="mlp", type=str,
        help="Architecture of the decoder, mlp with hidden_layers 0 is equivatent to linear")
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--decoder_features", default=None, type=int,
        help="Number of features on decoder hidden layers, ignored when decoder_layers is 0")
    parser.add_argument("--encoder_features", default=None, type=int)
    parser.add_argument("--decoder_layers", default=1, type=int)
    parser.add_argument("--mode", default="img", type=str,
        help="Mode of training, options: ts for time series as input for the model, img (default) for dissimilarity frames as input, dtw for dtw-layer encoding")
    parser.add_argument("--reduce_imbalance", action="store_true", 
        help="Wether to subsample imbalanced classes")
    parser.add_argument("--no-reduce_imbalance", dest="reduce_imbalance", action="store_false")
    parser.add_argument("--label_mode", default=1, type=int,
        help="Consider the mode (most common) label out of this number of labels for training (default 1), must be an odd number")
    parser.add_argument("--num_medoids", default=1, type=int,
        help="Number of medoids per class to use")
    parser.add_argument("--voting", default=1, type=int,
        help="Number of previous predictions to consider in the vote of the next prediction, defaults to 1 (no voting)")
    parser.add_argument("--rho", default=0.1, type=float,
        help="Parameter of the online-dtw algorithm, the window_size-th root is used as the voting parameter")
    parser.add_argument("--use_medoids", action="store_true", default=True,
        help="Use medoids for DM computation")
    parser.add_argument("--use_synthetic", action="store_false", dest="use_medoids", 
        help="Use synthetic shapes for DM computation")
    parser.add_argument("--overlap", default=-1, type=int, 
        help="Overlap of observations between training and test examples, default -1 for maximum overlap (equivalent to overlap set to window size -1)")
    parser.add_argument("--mtf_bins", default=10, type=int, 
        help="Number of bins for mtf computation")
    parser.add_argument("--training_dir", default="training", type=str, 
        help="Directory of model checkpoints")

    return parser

def load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize):
 
    if dataset_home_directory is None:
        dataset_home_directory = "./datasets"

    ds = None
    if dataset_name == "WISDM":
        ds = WISDMDataset(
            os.path.join(dataset_home_directory, dataset_name), 
            wsize=window_size, wstride=window_stride, normalize=normalize)
    elif dataset_name == "UCI-HAR":
        ds = UCI_HARDataset(
            os.path.join(dataset_home_directory, dataset_name), split="both", 
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=ucihar_label_mapping)
    elif dataset_name == "REALDISP":
        ds = REALDISPDataset(
            os.path.join(dataset_home_directory, dataset_name), sensor_position=["LLA", "BACK"], sensor=["ACC"], mode=["ideal"],
            wsize=window_size, wstride=window_stride, normalize=normalize)
    elif dataset_name == "HARTH":
        ds = HARTHDataset(
            os.path.join(dataset_home_directory, dataset_name), 
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=harth_label_mapping)
    elif dataset_name == "MHEALTH":
        ds = MHEALTHDataset(
            os.path.join(dataset_home_directory, dataset_name), sensor=["acc", "gyro"],
            wsize=window_size, wstride=window_stride, normalize=normalize)
    elif dataset_name == "HARTH-TESTS":
        ds = HARTHDataset(
            os.path.join(dataset_home_directory, "HARTH"), 
            wsize=window_size, wstride=window_stride, normalize=normalize, label_mapping=harth_label_mapping)
        ds.indices = ds.indices[ds.indices <= ds.subject_indices[2]] ## only first 2
        ds.splits = ds.splits[ds.splits <= ds.subject_indices[2]]
    
    return ds

def load_dmdataset(
        dataset_name,
        dataset_home_directory = None,
        rho = 0.1,
        batch_size = 16,
        num_workers = 1,
        window_size = 32,
        window_stride = 1,
        normalize = True,
        pattern_size = None,
        compute_n = 500,
        subjects_for_test = None,
        reduce_train_imbalance = False,
        num_medoids = 1,
        label_mode = 1,
        use_medoids = True,
        overlap = -1):
        
    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
        
    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {window_size}")

    # load medoids if already computed
    # if not os.path.exists(os.path.join(dataset_home_directory, dataset_name, f"meds{window_size}.npz")):
    #     print("Computing medoids...")
    #     meds = sts_medoids(ds, n=compute_n)
    #     with open(os.path.join(dataset_home_directory, dataset_name, f"meds{window_size}.npz"), "wb") as f:
    #         np.save(f, meds)
    # else:
    #     meds = np.load(os.path.join(dataset_home_directory, dataset_name, f"meds{window_size}.npz"))
    #     assert meds.shape[2] == pattern_size

    if use_medoids:
        print("Computing medoids...")
        meds = sts_medoids(ds, pattern_size=pattern_size, meds_per_class=num_medoids, n=compute_n)
    else:
        print("Using synthetic shapes...")
        meds = np.empty((3, pattern_size))
        meds[0,:] = np.linspace(-1, 1, pattern_size)
        meds[1,:] = np.linspace(1, -1, pattern_size)
        meds[2,:] = 0

    print("Computing dissimilarity frames...")
    dfds = DFDataset(ds, patterns=meds, rho=rho, dm_transform=None, cached=True, ram=False)

    data_split = split_by_test_subject(ds, subjects_for_test)

    if normalize:
        # get average values of the DM
        DM = []
        np.random.seed(42)
        for i in np.random.choice(np.arange(len(dfds))[data_split["train"](dfds.stsds.indices)], compute_n):
            dm, _, _ = dfds[i]
            DM.append(dm)
        DM = torch.stack(DM)

        dm_transform = Normalize(mean=DM.mean(dim=[0, 2, 3]), std=DM.std(dim=[0, 2, 3]))
        dfds.dm_transform = dm_transform

    dm = LDFDataset(dfds, data_split=data_split, batch_size=batch_size, random_seed=42, 
        num_workers=num_workers, reduce_train_imbalance=reduce_train_imbalance, label_mode=label_mode, overlap=overlap)

    print(f"Using {len(dm.ds_train)} observations for training and {len(dm.ds_val)} observations for validation and test")

    return dm


def load_tsdataset(
        dataset_name,
        dataset_home_directory = None,
        batch_size = 16,
        num_workers = 1,
        window_size = 32,
        window_stride = 1,
        normalize = True,
        pattern_size = None,
        subjects_for_test = None,
        reduce_train_imbalance = False,
        label_mode = 1,
        overlap = -1,
        mode = None,
        mtf_bins = 50):
    
    ds = load_dataset(dataset_name, dataset_home_directory, window_size, window_stride, normalize)
        
    print(f"Loaded dataset {dataset_name} with a total of {len(ds)} observations for window size {window_size}")

    data_split = split_by_test_subject(ds, subjects_for_test)

    dm = LSTSDataset(ds, data_split=data_split, batch_size=batch_size, random_seed=42, 
        num_workers=num_workers, reduce_train_imbalance=reduce_train_imbalance, label_mode=label_mode, overlap=overlap, mode=mode, mtf_bins=mtf_bins)
    dm.l_patterns = pattern_size

    print(f"Using {len(dm.ds_train)} observations for training and {len(dm.ds_val)} observations for validation and test")

    return dm

def load_dm(args):
    if args.mode == "img":
        dm = load_dmdataset(
            args.dataset, dataset_home_directory=args.dataset_dir, 
            rho=args.rho, batch_size=args.batch_size, num_workers=args.num_workers, 
            window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size, 
            compute_n=args.compute_n, subjects_for_test=args.subjects_for_test, reduce_train_imbalance=args.reduce_imbalance, 
            label_mode=args.label_mode, num_medoids=args.num_medoids, use_medoids=args.use_medoids, overlap=args.overlap)
    elif args.mode in ["ts", "dtw", "dtw_c", "mtf", "gasf", "gadf"]:
        dm = load_tsdataset(
            args.dataset, dataset_home_directory=args.dataset_dir, 
            batch_size=args.batch_size, num_workers=args.num_workers, 
            window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size,
            subjects_for_test=args.subjects_for_test, reduce_train_imbalance=args.reduce_imbalance, 
            label_mode=args.label_mode, overlap=args.overlap, mode=args.mode, mtf_bins=args.mtf_bins)
        
    return dm

def get_model(name, args, dm):
    model = create_model_from_DM(dm, name=name, 
        dsrc=args.mode, arch=args.encoder_architecture, dec_arch=args.decoder_architecture,
        task="cls", lr=args.lr, enc_feats=args.encoder_features, 
        dec_feats=args.decoder_features, dec_layers=args.decoder_layers,
        voting={"n": args.voting, "rho": args.rho})
    
    return model

def save_csv(args, data, root_dir, filename):
    filepath = os.path.join(root_dir, filename)
    if os.path.exists(filepath):
        os.rename(filepath, filepath + "old")
        df = pd.read_csv(filepath + "old")
    elif os.path.exists(filepath + "old"):
        a = time()
        while time()-a < 10: # wait max of 10 seconds if another process is accessing the csv
            if os.path.exists(filepath):
                os.rename(filepath, filepath + "old")
                df = pd.read_csv(filepath + "old")
                break
    else:
        df = pd.DataFrame()

    save_data = {**data, **args.__dict__}
    for key in save_data.keys():
        if key not in df.columns:
            df[key] = None
    df.loc[len(df)] = save_data

    df.to_csv(filepath)

    if os.path.exists(filepath + "old"):
        os.remove(filepath + "old")