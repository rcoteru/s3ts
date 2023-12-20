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

    modelname = get_model_name(args)

    model = create_model_from_DM(dm, name=modelname, 
        dsrc=args.mode, arch=args.encoder_architecture, dec_arch=args.decoder_architecture,
        task="cls", lr=args.lr, enc_feats=args.encoder_features, 
        dec_feats=args.decoder_features, dec_layers=args.decoder_layers,
        voting={"n": args.voting, "rho": args.rho})
    
    model, data = train_model(dm, model, max_epochs=args.max_epochs)
    print(data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")