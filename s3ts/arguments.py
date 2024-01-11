
def get_model_name(args):
    modelname = f"{args.mode}|{args.dataset}," + '-'.join([str(subject) for subject in args.subjects_for_test]) + "|" \
                f"{args.window_size},{args.window_stride}|bs{args.batch_size}_lr{args.lr}|" + \
                f"{args.encoder_architecture}{args.encoder_features}|" + \
                f"{args.decoder_architecture}{args.decoder_features},{args.decoder_layers}|" + \
                (f"m{args.label_mode}|" if args.label_mode > 1 else "") + \
                (f"v{args.voting},{args.rho}|" if args.voting > 1 else "") + \
                (f"ov{args.overlap}|" if args.overlap > 0 else "")
    
    if args.mode in ["img", "dtw", "dtw_c"]:
        modelname += f"p{args.pattern_size},r{args.rho}|"
    if args.mode == "img":
        modelname += f"med{args.num_medoids},{args.compute_n}|" if args.use_medoids else f"syn{args.compute_n}|"
    if args.mode == "mtf":
        modelname += f"bin{args.mtf_bins}|"

    return modelname[:-1]

def get_command(args):
    command = f"--mode {args.mode} --dataset {args.dataset} --lr {args.lr} " + \
                "--subjects_for_test " + ' '.join([str(subject) for subject in args.subjects_for_test]) + " " \
                f"--window_size {args.window_size} --window_stride {args.window_stride} --batch_size {args.batch_size} " + \
                f"--encoder_architecture {args.encoder_architecture} --encoder_features {args.encoder_features} " + \
                f"--decoder_architecture {args.decoder_architecture} --decoder_features {args.decoder_features} --decoder_layers {args.decoder_layers} " + \
                (f"--label_mode {args.label_mode} " if args.label_mode > 1 else "") + \
                (f"--voting {args.voting} --rho {args.rho} " if args.voting > 1 else "") + \
                (f"--overlap {args.overlap} " if args.overlap > 0 else "")
    
    if args.mode in ["img", "dtw", "dtw_c"]:
        command += f"--pattern_size {args.pattern_size} --rho {args.rho} "
    if args.mode == "img":
        command += f"--compute_n {args.compute_n} "
        command += f"--use_medoids --num_medoids {args.num_medoids} " if args.use_medoids else f"--use_synthetic "
    if args.mode == "mtf":
        command += f"--mtf_bins {args.mtf_bins} "

    command += f"--num_workers {args.num_workers} --max_epochs {args.max_epochs} --normalize --reduce_imbalance "
    command += f"--training_dir {args.training_dir}"

    return command