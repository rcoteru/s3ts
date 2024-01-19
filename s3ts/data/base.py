import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset


class StreamingFramesDM(LightningDataModule):

    """ Abstract class for datamodules. """

    # these properties are used to automagically configure models,
    # so they should be set in the constructor

    wdw_len: int        # window length
    wdw_str: int        # window stride
    sts_str: bool       # stride the series too?

    n_dims: int         # number of STS dimensions
    n_classes: int      # number of classes
    n_patterns: int     # number of patterns
    l_patterns: int     # pattern size

    batch_size: int     # dataloader batch size
    random_seed: int    # random seed
    num_workers: int    # dataloader nworkers

    ds_train: Dataset   # train dataset
    ds_val: Dataset     # validation dataset
    ds_test: Dataset    # test dataset

    # datasets should return a dict for each sample
    # {"frame": torch.Tensor, "series": torch.Tensor, "label": torch.Tensor}
    # where:
    # - "frame" is the streaming frame
    # - "series" is the TS for the regression task
    # - "label" is the label of the frame (integer form, not one-hot)


class STSDataset(Dataset):

    def __init__(self,
            wsize: int = 10,
            wstride: int = 1,
            ) -> None:
        super().__init__()

        '''
            Base class for STS dataset

            Inputs:
                wsize: window size
                wstride: window stride
        '''

        self.wsize = wsize
        self.wstride = wstride

        self.splits = None

        self.STS = None
        self.SCS = None

        self.indices = None

    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:

        first = self.indices[index]-self.wsize*self.wstride+1
        last = self.indices[index]+1

        return self.STS[:, first:last:self.wstride], self.SCS[first:last:self.wstride]
    
    def sliceFromArrayOfIndices(self, indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(indexes.shape) == 1 # only accept 1-dimensional arrays

        return_sts = np.empty((indexes.shape[0], self.STS.shape[0], self.wsize))
        return_scs = np.empty((indexes.shape[0], self.wsize))

        for i, id in enumerate(indexes):
            ts, c = self[id]
            return_scs[i] = c
            return_sts[i] = ts

        return return_sts, return_scs
    
    def getSameClassWindowIndex(self):

        id = []
        cl = []
        for i, ix in enumerate(self.indices):
            if np.unique(self.SCS[(ix-self.wsize*self.wstride):ix]).shape[0] == 1:
                id.append(i)
                cl.append(self.SCS[ix])
        
        return np.array(id), np.array(cl)
    
    def normalizeSTS(self, mode):
        self.mean = np.expand_dims(self.STS.mean(1), 1)
        self.std = np.expand_dims(np.std(self.STS, axis=1), 1)

        self.STS = (self.STS - self.mean) / self.std
