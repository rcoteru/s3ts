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


class ResNet_Encoder(LightningModule):

    def __init__(self, ref_size: int , channels: int, window_size: int):
        super().__init__()

        self.channels = channels
        self.n_feature_maps = 32

        self.model = nn.Sequential(
            ResidualBlock(in_channels=channels, out_channels=self.n_feature_maps),
            ResidualBlock(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2),
            ResidualBlock(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2),
            ResidualBlock(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 4),
            nn.AvgPool2d((ref_size, window_size))
        )

    @staticmethod
    def __str__() -> str:
        return "ResNet"

    def get_output_shape(self):
        return (len(self.model._modules) -1) * self.n_feature_maps

    def forward(self, x):
        return self.model(x.float())

    