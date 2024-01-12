baseArguments = {
    "num_workers": 8,
    "dataset": "HARTH",
    "subjects_for_test": [
        [21]
    ],
    "lr": 0.001,
    "n_val_subjects": 5,
    "encoder_architecture": "cnn_gap",
    "encoder_features": 20,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "batch_size": 128,
    "label_mode": 1,
    "voting": 1,
    "rho": 0.1,
    "overlap": -1,
    "max_epochs": 15,
    "training_dir": "training_new"
}

imgExperiments = {
    "window_size": [16, 32, 48],
    "window_stride": [1, 2, 4],
    "mode": "img",
    "num_medoids": 1,
    "compute_n": 300,
    "use_medoids": [True, False],
    "pattern_size": [32, 48]
}

dtwExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "dtw",
    "pattern_size": [8, 16, 24]
}

dtwcExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "dtw_c",
    "pattern_size": [8, 16, 24]
}

tsExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "ts"
}

gasfExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "gasf"
}

gadfExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "gadf"
}

mtffExperiments = {
    "window_size": 48,
    "window_stride": 1,
    "mode": "mtf",
    "mtf_bins": 10
}

RAM = 32
CPUS = 16

experiments = [imgExperiments, dtwExperiments, dtwcExperiments, tsExperiments, gasfExperiments, gadfExperiments, mtffExperiments]
