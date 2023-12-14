import torch
from pytorch_lightning import LightningModule
from torch import nn

class CNN_TS(LightningModule):

    """ Basic CNN for time series. """

    def __init__(self, channels: int, wdw_size: int, ref_size: int,
                 n_feature_maps: int = 32):
        super().__init__()

        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = 1 # here for compatibility
        self.n_feature_maps = n_feature_maps

        # convolutional layer 0
        self.cnn_0 = nn.Sequential(nn.Conv1d(in_channels=channels, 
            out_channels=self.n_feature_maps//2, kernel_size=3, padding='same'),
            nn.ReLU(), nn.MaxPool1d(kernel_size=2))
        
        # convolutional layer 1
        self.cnn_1 = nn.Sequential(nn.Conv1d(in_channels=self.n_feature_maps//2, 
            out_channels=self.n_feature_maps, kernel_size=3, padding='same'),
            nn.ReLU(), nn.AvgPool1d(kernel_size=2), nn.Dropout(0.35))
        
        # convolutional layer 2
        self.cnn_2 = nn.Sequential(nn.Conv1d(in_channels=self.n_feature_maps, 
            out_channels=self.n_feature_maps*2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=self.n_feature_maps*2),
            nn.ReLU(), nn.Dropout(0.4))

        # convolutional layer 3
        self.cnn_3 = nn.Conv1d(in_channels=self.n_feature_maps*2, 
            out_channels=self.n_feature_maps*4, kernel_size=3, padding='same')
            
    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape
    
    def forward(self, x) -> torch.Tensor:
        feats = self.cnn_0(x.float())
        feats = self.cnn_1(feats)
        feats = self.cnn_2(feats)
        feats = self.cnn_3(feats)
        return feats