# Self-Supervised Streaming Time Series Classification

Time-series classification though convolutional neural networks, where the time-series data is converted to images through an online, DTW-based (Dynamic Time Warping) similarity measure. Additional tasks are used during training to improve generalization, derived only from the original dataset without any additional data. 

The tasks include:

 - Classification of a discretized version of the time-series.
 - Prediction of the time series, either regression or classification.
 - Convolutional autoencoder of the sliding frames. (TODO)

## Environment / Setup

```bash
git clone https://github.com/rcote98/s3ts.git   # clone the repo
cd s3ts                                         # move in the folder
python3 -m venv s3ts_env                        # create virtualenv
source s3ts_env/bin/activate                    # activate it
pip install -r requirements.txt                 # install dependencies
python -m pip install -e .                      # install dev package
```

### Things to check
- Influence of shifting labels in the pretrain
- Performance with ratios of train/pretrain

### Ideas
- Use [learned shapelets](https://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf) instead of medoids as patterns.
- Use artificial patterns alongside the medoids
- Use only artifical patterns -> zero-shot learning!
- Include data augmentation techniques


### TODO's
- Generalize model training / saving
- Gather training stats automatically
- Do some nice plotting
- Multiseed checks
