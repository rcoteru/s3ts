import torch
import numpy as np

def getAUC(p, c, c0):
    '''
    p: probabilidad de la clase, p(C|x)
    c: array de clases reales (c,x)
    c0: la clase (positiva) para la que calculamos el AUC (int)
    '''
    c01 = np.array(c != c0, dtype=np.int64)
    p0 = p[:, c0]

    (N0, N1) = np.unique(c01, return_counts=True)[1]
    delta = N0
    tau = 0

    sort = np.argsort(-p0)
    for i in range(len(sort)):
        if (c01[sort[i]] == 0):  # cuando es un cero
            delta -= 1
        else:  # cuando es un 1
            tau += delta
    AUC = 1.0 - (tau / (N0 * N1))
    return AUC


def getAvgAUC(p, c):
    return np.average([getAUC(p, c, c0) for c0 in range(p.shape[1])])


def torchAUROC(p, c, num_classes):
    c01 = torch.eye(num_classes, dtype=torch.bool)[c]
    psort, sortindices = torch.sort(p, dim=0, descending=True)
    c01sort = torch.gather(c01, 0, sortindices)

    N0 = torch.sum(c01, dim=0) # equal to class
    N1 = c01.shape[0] - N0 # diff to class

    delta = N0 - torch.cumsum(c01sort, dim=0)
    tau = (delta * ~c01sort).sum(dim=0)

    return 1.0 - (tau / (N0 * N1))