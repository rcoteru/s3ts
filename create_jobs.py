import os

from s3ts.arguments import get_model_name, get_command

baseArguments = {
    "num_workers": 8,
    "dataset": "HARTH",
    "subjects_for_test": [
        [21]
    ],
    "lr": 0.001,
    "encoder_architecture": "cnn_gap",
    "encoder_features": 20,
    "decoder_architecture": "mlp",
    "decoder_features": 32,
    "decoder_layers": 1,
    "window_size": 48,
    "window_stride": 1,
    "batch_size": 128,
    "label_mode": 1,
    "voting": 1,
    "rho": 0.1,
    "overlap": -1,
    "max_epochs": 30
}

imgExperiments = {
    "mode": "img",
    "num_medoids": 1,
    "compute_n": 300,
    "use_medoids": [True, False],
    "pattern_size": [16, 32, 48]
}

dtwExperiments = {
    "mode": "dtw",
    "pattern_size": [8, 16, 24]
}

dtwcExperiments = {
    "mode": "dtw_c",
    "pattern_size": [8, 16, 24]
}

tsExperiments = {
    "mode": "ts"
}

gasfExperiments = {
    "mode": "gasf"
}

gadfExperiments = {
    "mode": "gadf"
}

mtffExperiments = {
    "mode": "mtf",
    "mtf_bins": 10
}

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

python training.py {get_command(args)}
'''

class EmptyExperiment:
    pass

def produce_experiments(args):

    print("Experiments for mode", args["mode"])

    multiple_arguments = []
    for key, value in args.items():
        if "__" not in key:
            if isinstance(value, list):
                multiple_arguments.append((key, len(value)))

    total_experiments = 1
    for i in multiple_arguments:
        total_experiments *= i[1]

    experiment_arguments = [EmptyExperiment() for i in range(total_experiments)]

    for exp in experiment_arguments:
        exp.__dict__.update(args)

    k = 1
    for key, value in args.items():
        if "__" in key:
            continue

        if isinstance(value, list):
            n = len(value)
            if n>1:
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
        jobname = jobname.replace("|", "_")
        jobname = jobname.replace(",", "_")
        jobs.append("sbatch " + jobname + ".job")
        with open(os.path.join("./", "cache_jobs", jobname + ".job"), "w") as f:
            f.write(job)

    print("Created", len(jobs), "jobs")

    return jobs

if __name__ == "__main__":
    if not os.path.exists("cache_jobs"):
        os.mkdir(os.path.join("./", "cache_jobs"))

    jobs = []
    for exp in [imgExperiments, dtwExperiments, dtwcExperiments, tsExperiments, gasfExperiments, gadfExperiments, mtffExperiments]:
        jobs += produce_experiments({**baseArguments, **exp})
    
    bash_script = "#!\\bin\\bash\n" + "\n".join(jobs)
    bash_script = bash_script.replace("|", "_")
    bash_script = bash_script.replace(",", "_")
    
    with open(os.path.join("./", "cache_jobs", "launch.sh"), "w") as f:
        f.write(bash_script)

    print(f"Number of experiments created: {len(jobs)}")
