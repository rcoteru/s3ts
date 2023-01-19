# Self-Supervised Streaming Time Series Classification

Time-series classification though convolutional neural networks, where the time-series data is converted to images through an online, DTW-based (Dynamic Time Warping) similarity measure. 

## Environment / Setup

```bash
git clone https://github.com/rcote98/s3ts.git   # clone the repo
cd s3ts                                         # move in the folder
python3 -m venv s3ts_env                        # create virtualenv
source s3ts_env/bin/activate                    # activate it
pip install -r requirements.txt                 # install dependencies
python -m pip install -e .                      # install dev package
```

## Visualize Training Progress
```bash
tensorboard --logdir=data/[whatever-dir-name]/lightning_logs/
```