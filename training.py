from s3ts.helper_functions import *
import multiprocessing

from s3ts.api.nets.methods import create_model_from_DM, train_model

from argparse import ArgumentParser

def main(args):

    if args.mode == "img":
        dm = load_dmdataset(
            args.dataset, dataset_home_directory=args.dataset_dir, 
            rho=args.rho, batch_size=args.batch_size, num_workers=args.num_workers, 
            window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size, 
            compute_n=args.compute_n, subjects_for_test=args.subjects_for_test, reduce_train_imbalance=args.reduce_imbalance, 
            label_mode=args.label_mode, num_medoids=args.num_medoids, use_medoids=args.use_medoids, overlap=args.overlap)
    elif args.mode in ["ts", "dtw"]:
        dm = load_tsdataset(
            args.dataset, dataset_home_directory=args.dataset_dir, 
            batch_size=args.batch_size, num_workers=args.num_workers, 
            window_size=args.window_size, window_stride=args.window_stride, normalize=args.normalize, pattern_size=args.pattern_size,
            subjects_for_test=args.subjects_for_test, reduce_train_imbalance=args.reduce_imbalance, 
            label_mode=args.label_mode, overlap=args.overlap)

    modelname = f"{'med' if args.use_medoids else 'syn'}_{args.dataset}_{args.mode}_rho{args.rho}_lr{args.lr}_bs{args.batch_size}_" + \
                f"{args.encoder_architecture}{args.encoder_features}_" + \
                f"{args.decoder_architecture}{args.decoder_features}_{args.decoder_layers}_" + \
                f"w{args.window_size}.{args.window_stride}_p{args.pattern_size}_" + \
                f"lmode{args.label_mode}_v{args.voting}_ovrlp{args.overlap}"

    model = create_model_from_DM(dm, name=modelname, 
        dsrc=args.mode, arch=args.encoder_architecture, dec_arch=args.decoder_architecture,
        task="cls", lr=args.lr, enc_feats=args.encoder_features, 
        dec_feats=args.decoder_features, dec_layers=args.decoder_layers,
        voting={"n": args.voting, "rho": args.rho})
    
    model, data = train_model(dm, model, max_epochs=args.max_epochs)
    print(data)

if __name__ == "__main__":
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

    args = parser.parse_args()
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")