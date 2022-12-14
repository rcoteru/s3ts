# Self-Supervised Streaming Time Series Classification

Time-series classification though convolutional neural networks, where the time-series data is converted to images through an online, DTW-based (Dynamic Time Warping) similarity measure. Additional tasks are used during training to improve generalization, derived only from the original dataset without any additional data. 

The tasks include:

 - Classification of a discretized version of the time-series.
 - Prediction of the time series, either regression or classification.
 - Regression of the original time series.
 - Regression of the sliding frames.

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