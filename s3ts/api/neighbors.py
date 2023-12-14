#/usr/bin/python3
# -*- coding: utf-8 -*-

""" 
    KNNeighbors Classifier with DTW metric.
"""

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score, f1_score
from s3ts.legacy.modules import DFDataModule
from tqdm import tqdm
import numpy as np

def knn_dtw_predict(dm: DFDataModule, metric = "dtw", n_neighbors = 1) -> tuple[float, float, KNeighborsTimeSeriesClassifier]:

    """ 
    Train a KNN classifier with DTW metric and predict on the test set.
    
    Parameters
    ----------
    dm : DFDataModule
        Data module with the train and test sets.
    metric : str, optional
        Distance metric to use. The default is "dtw".
    n_neighbors : int, optional
        Number of neighbors to use. The default is 1.
    
    Returns
    -------
    acc : float
        Accuracy score on the test set.
    f1 : float
        F1 score on the test set.
    model : KNeighborsTimeSeriesClassifier
        Trained model.
    """
    
    model = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors, distance=metric)
    X_train = dm.X_train.numpy()
    Y_train = dm.Y_train.numpy()
    model.fit(X_train, Y_train)

    Y_true, Y_pred = [], []
    print("Predicting on the test set...")
    for batch in tqdm(dm.test_dataloader()):
        series: np.ndarray = batch[1].numpy()
        Y_pred.append(model.predict(series))
        Y_true.append(batch[2].argmax(axis=1).numpy())

    # join the batches
    Y_pred, Y_true = np.concatenate(Y_pred), np.concatenate(Y_true)

    # calculate metrics
    acc = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, average="micro")

    return acc, f1, model