"""
Kind obvious tbh.

@author Raúl Coterillo
@version 2023-01
"""

# data
from s3ts.models.encoders.ResNet import ResNet_Encoder
from s3ts.models.encoders.CNN import CNN_Encoder
from s3ts.setup.pred import compare_pretrain

from pathlib import Path
import pandas as pd

# SETTINGS
# =================================

DIR = Path("test")

DATASET = "GunPoint"
PRETRAIN = 1
ENCODER = CNN_Encoder
LAB_SHIFTS = [0]

RANDOM_STATE = 0
RANDOM_STATE_TEST = 0

