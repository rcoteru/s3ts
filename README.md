# Self-Supervised Streaming Time Series 

## Environment / Setup


```bash
git clone https://github.com/rcote98/s3ts.git   # clone the repo
cd s3ts                                         # move in the folder
python3 -m venv s3ts_env                        # create virtualenv
source env/bin/activate                         # activate it
pip install -r requirements.txt                 # install dependencies
python -m pip install -e .                      # install dev package
```

## Things to check
- Influence of shifting labels in the pretrain
- Performance with ratios of train/pretrain

## Ideas
- Use artificial patterns alongside the good ones
- Include data augmentation techniques

## TODO
- Generalize model training / saving
- Gather training stats automatically
- Do some nice plotting
- Multiseed checks
