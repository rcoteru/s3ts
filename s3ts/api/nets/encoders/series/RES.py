import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

class ResidualBlock_1d(LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm1d(out_channels)
        )

        self.block.apply(ResidualBlock_1d.initialize_weights)
        self.shortcut.apply(ResidualBlock_1d.initialize_weights)

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
        elif hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        block = self.block(x.float())
        shortcut = self.shortcut(x.float())
        block = torch.add(block, shortcut)
        return F.relu(block)


class RES_TS(LightningModule):

    """ Residual CNN for times series. """

    def __init__(self, channels: int, wdw_size: int, ref_size: int,
                 n_feature_maps: int = 32):
        super().__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = 1 # here for compatibility
        self.n_feature_maps = n_feature_maps

        self.res_0 = ResidualBlock_1d(in_channels=channels, out_channels=self.n_feature_maps)
        self.res_1 = ResidualBlock_1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2)
        self.res_2 = ResidualBlock_1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2)
        self.res_3 = ResidualBlock_1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 4)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape
    
    def forward(self, x):
        feats = self.res_0(x.float())
        feats = self.res_1(feats)
        feats = self.res_2(feats)
        feats = self.res_3(feats)
        feats = self.pool(feats)  
        return feats