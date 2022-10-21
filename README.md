# Self-Supervised Streaming TIme Series 

## Environment / Setup

Clone repo and move into folder
```bash
git clone https://github.com/rcote98/s3ts.git
cd s3ts
```

Create virtual envirnment
```bash
python3 -m venv env
```

Activate environment
```bash
source env/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Install _s3ts_ in development mode
```bash
python -m pip install -e .
```

## Usage

Download dataset from repository and prepare it.
```bash
python3 scripts/prepare_data.py
```

Train neural network
```bash
python3 scripts/train_network.py
```
