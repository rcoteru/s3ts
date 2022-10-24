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

## Usage / Scripts

Download dataset from repository and prepare it.
```bash
python3 scripts/prepare_data.py
```

Train neural network
```bash
python3 scripts/train_network.py
```

## TODO

- Improve training set creation
- Generalize task scheduling
- Generalize model training / saving
- Use longer STSs instead of more STSs / drop beggining of STS
- Do some nice plots
