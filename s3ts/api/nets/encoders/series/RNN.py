import torch
from pytorch_lightning import LightningModule
from torch import nn


class RNN_TS(LightningModule):

    """ Recurrent neural network (LSTM) for times series. """

    def __init__(self, channels: int, wdw_size: int, ref_size: int,
                 n_feature_maps: int = 32):
        super().__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = 1 # here for compatibility
        self.n_feature_maps = n_feature_maps

        self.lstm_1 = nn.LSTM(input_size=self.channels, hidden_size=self.n_feature_maps, dropout=0.2, batch_first=True)
        self.bn_1 = nn.BatchNorm1d(num_features=self.n_feature_maps, momentum=0.999, eps=0.01)
        self.lstm_2 = nn.LSTM(input_size=self.n_feature_maps, hidden_size=self.n_feature_maps * 2, dropout=0.2, batch_first=True)
        self.bn_2 = nn.BatchNorm1d(num_features=self.n_feature_maps * 2, momentum=0.999, eps=0.01)

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = torch.permute(x.float(), (0, 2, 1))
        output, _ = self.lstm_1(x)
        output: torch.Tensor
        ret_bn1: torch.Tensor = self.bn_1(output.permute(0, 2, 1))
        output, _ = self.lstm_2(ret_bn1.permute(0, 2, 1))
        ret_bn2: torch.Tensor = self.bn_2(output.permute(0, 2, 1))
        return ret_bn2.view(ret_bn2.shape[0], -1)