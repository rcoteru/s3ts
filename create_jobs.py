import os

dataset = "WISDM"
subjects_for_test = [30, 31, 32, 33, 34, 35]
epochs = 10

BATCH_SIZES = [128]
LEARNING_RATES = [1e-4]
ENCODERS = ["simplecnn"]
ENCODER_FEATURES = [4]
DECODERS = ["mlp"]
MODES = ["dtw", "img", "ts"]
DECODER_FEATURES = [16]
WINDOW_SIZES = [20]
WINDOW_STRIDES = [1, 2]
DECODER_LAYERS = 1
RAM = 32

def create_jobs(mode, batch_size, window_size, window_stride, learning_rate, encoder, encoder_features, decoder, decoder_features):

    jobname = f"job_{mode}_{encoder}{encoder_features}_{decoder}{decoder_features}_{DECODER_LAYERS}" + \
              f"_lr{learning_rate}_wsize{window_size}_wstride{window_stride}_bs{batch_size}"
    return jobname, f'''#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem={RAM}GB
#SBATCH --time=10:00:00
#SBATCH --job-name={jobname}
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

cd $HOME/s3ts

source $HOME/.bashrc
source activate dev

python training.py --dataset {dataset} --window_size {window_size*2 if mode=="dtw" else window_size} --window_stride {window_stride} \\
--pattern_size {window_size} \\
--subjects_for_test {" ".join([str(subject) for subject in subjects_for_test])} \\
--encoder_architecture {encoder} --encoder_features {encoder_features} \\
--decoder_architecture {decoder} --decoder_features {decoder_features} --decoder_layers {DECODER_LAYERS} \\
--mode {mode} \\
--batch_size {batch_size} --lr {learning_rate} --num_workers 8
'''

if __name__ == "__main__":
    if not os.path.exists("cache_jobs"):
        os.mkdir(os.path.join("./", "cache_jobs"))

    jobs = []

    for mode in MODES:
        for batch_size in BATCH_SIZES:
            for window_size in WINDOW_SIZES:
                for window_stride in WINDOW_STRIDES:
                    for learning_rate in LEARNING_RATES:
                        for encoder in ENCODERS:
                            for encoder_features in ENCODER_FEATURES:
                                for decoder in DECODERS:
                                    for decoder_features in DECODER_FEATURES:
                                        jobname, job = create_jobs(mode, batch_size, window_size, window_stride, learning_rate, encoder, encoder_features, decoder, decoder_features)
                                        
                                        jobs.append("sbatch " + jobname + ".job")
                                        with open(os.path.join("./", "cache_jobs", jobname + ".job"), "w") as f:
                                            f.write(job)

    bash_script = "#!\\bin\\bash\n" + "\n".join(jobs)
    with open(os.path.join("./", "cache_jobs", "launch.sh"), "w") as f:
        f.write(bash_script)
