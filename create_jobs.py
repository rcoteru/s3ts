import os
import numpy as np

from s3ts.helper_functions import get_model_name, get_parser

class Experiments:
    num_workers = 8
    dataset = "HARTH"
    subjects_for_test = [
        [21]
    ]
    epochs = 30
    rho = 0.1
    batch_size = 128
    lr = 0.001
    mode = "img"
    encoder_architecture = "cnn_gap"
    encoder_features = 20
    decoder_architecture = "mlp"
    decoder_features = 32
    decoder_layers = 1
    window_size = 48
    window_stride = [1, 2]
    num_medoids = 1
    compute_n = 300
    overlap = [-1, 45]
    use_medoids = [True, False]
    label_mode = 1
    voting = 1
    pattern_size = [16, 32, 48]

def create_jobs(args):
    modelname = get_model_name(args)

    return modelname, f'''#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --job-name={modelname}
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

cd $HOME/s3ts

source $HOME/.bashrc
source activate dev

python training.py --dataset {args.dataset} --window_size {args.window_size*2 if args.mode=="dtw" else args.window_size} --window_stride {args.window_stride} \\
--pattern_size {args.pattern_size} \\
--subjects_for_test {" ".join([str(subject) for subject in args.subjects_for_test])} \\
--encoder_architecture {args.encoder_architecture} --encoder_features {args.encoder_features} \\
--decoder_architecture {args.decoder_architecture} --decoder_features {args.decoder_features} --decoder_layers {args.decoder_layers} \\
--mode {args.mode} --max_epochs {args.epochs} \\
--batch_size {args.batch_size} --lr {args.lr} --num_workers {args.num_workers} \\
--reduce_imbalance --normalize --label_mode {args.label_mode} --num_medoids {args.num_medoids} --compute_n {args.compute_n} \\
--rho {args.rho} --voting {args.voting} --use_{"medoids" if args.use_medoids else "synthetic"} --overlap {args.overlap}
'''

if __name__ == "__main__":
    if not os.path.exists("cache_jobs"):
        os.mkdir(os.path.join("./", "cache_jobs"))

    multiple_arguments = []
    for key, value in Experiments.__dict__.items():
        if "__" not in key:
            if isinstance(value, list):
                multiple_arguments.append((key, len(value)))

    total_experiments = np.prod([it[1] for it in multiple_arguments])
    total_experiments

    parser = get_parser()
    experiment_arguments = [parser.parse_args([]) for i in range(total_experiments)]

    k = 1
    for key, value in Experiments.__dict__.items():
        if "__" in key:
            continue

        if isinstance(value, list):
            n = len(value)
            print("Argument with multiple values:", key, "with", n, "values")
            for i, experiment_arg in enumerate(experiment_arguments):
                setattr(experiment_arg, key, value[(i//k)%n])
            k *= n
        else:
            for experiment_arg in experiment_arguments:
                setattr(experiment_arg, key, value)
    
    jobs = []

    for exp_arg in experiment_arguments:
        jobname, job = create_jobs(exp_arg)
        jobs.append("sbatch " + jobname + ".job")
        with open(os.path.join("./", "cache_jobs", jobname + ".job"), "w") as f:
            f.write(job)

    bash_script = "#!\\bin\\bash\n" + "\n".join(jobs)
    with open(os.path.join("./", "cache_jobs", "launch.sh"), "w") as f:
        f.write(bash_script)

    print(f"Number of experiments created: {len(jobs)}")
