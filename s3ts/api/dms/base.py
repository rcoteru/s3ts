from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

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