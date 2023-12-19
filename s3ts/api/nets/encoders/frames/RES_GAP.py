#lightning
from pytorch_lightning import LightningModule

from torch import nn
import torch

class ResidualBlock(LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=8, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm2d(out_channels)
        )
        self.block.apply(ResidualBlock.initialize_weights)
        self.shortcut.apply(ResidualBlock.initialize_weights)

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
        elif hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor):
        block = self.block(x)
        shortcut = self.shortcut(x)
        block = torch.add(block, shortcut)
        return nn.functional.relu(block)

class RES_GAP_IMG(LightningModule):

    def __init__(self, channels: int, wdw_size: int, ref_size: int,
                 n_feature_maps: int = 32):
        super().__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = ref_size
        self.n_feature_maps = n_feature_maps

        self.res_0 = ResidualBlock(in_channels=channels, out_channels=self.n_feature_maps)
        self.res_1 = ResidualBlock(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps)
        self.res_2 = ResidualBlock(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps)
        
    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.ref_size, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape


    def forward(self, x):
        feats = self.res_0(x.float())
        feats = self.res_1(feats)
        feats = self.res_2(feats) 
        return feats.mean(dim=(-2, -1))

    