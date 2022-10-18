
# data processing
from calendar import c
from s3ts.datasets.processing import acquire_dataset, compute_medoids, build_STSs
from s3ts.datasets.processing import discretize_STSs, shift_STSs_labels
from s3ts.datasets.oesm import compute_ESMs

# convolutional network 

# plotting
import matplotlib.pyplot as plt

# Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

STS_NUMBER = 40
STS_LENGTH = 10

BIN_NUMBER = 5
SAMPLE_SHFT = -2

RHO = 0.1

PLOTS = False
NPROCS = 4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# grab the data
X, Y, mapping = acquire_dataset("GunPoint")

# compute medoids 
medoids, medoid_ids = compute_medoids(X, Y, distance_type="dtw")

# plot the medoids
if PLOTS:
    for i in range(medoids.shape[0]):
        plt.plot(medoids[i])
        plt.title(f"Medoid {i}")
        plt.show()

# create STSs from samples
STSs_X, STSs_Y = build_STSs(X=X, Y=Y, samples_per_sts=STS_LENGTH, 
                        number_of_sts=STS_NUMBER, skip_ids=medoid_ids)   

# discretize each time series to obtain new labels
STSs_Yd = discretize_STSs(STSs_X, intervals=BIN_NUMBER, strategy="quantile")

# shift the labels
STSs_Xs, STSs_Yds = shift_STSs_labels(STSs_X, STSs_Yd, SAMPLE_SHFT)

# compute the ESMs
ESMs = compute_ESMs(STSs_X, medoids, rho=RHO, nprocs=NPROCS)
#ESMs_shift = compute_ESMs(STSs_Xs, )