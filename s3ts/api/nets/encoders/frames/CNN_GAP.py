import torch
import torch.nn as nn

class CNN_GAP_IMG(torch.nn.Module):
    
    def __init__(self, channels=1, ref_size=32, 
            wdw_size=32, n_feature_maps=32):
        super().__init__()
        
        # register parameters
        self.channels = channels
        self.wdw_size = wdw_size
        self.ref_size = ref_size
        self.n_feature_maps = n_feature_maps

        # convolutional layer 0
        self.cnn_0 = nn.Sequential(nn.Conv2d(in_channels=channels, 
            out_channels=self.n_feature_maps, kernel_size=8, padding='same'),
            nn.BatchNorm2d(num_features=self.n_feature_maps),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        
        # convolutional layer 1
        self.cnn_1 = nn.Sequential(nn.Conv2d(in_channels=self.n_feature_maps, 
            out_channels=self.n_feature_maps*2, kernel_size=5, padding='same'),
            nn.BatchNorm2d(num_features=self.n_feature_maps*2),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        
        # convolutional layer 2
        self.cnn_2 = nn.Sequential(nn.Conv2d(in_channels=self.n_feature_maps*2, 
            out_channels=self.n_feature_maps, kernel_size=3, padding='valid'),
            nn.BatchNorm2d(num_features=self.n_feature_maps),
            nn.ReLU())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_0(x)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        return x.mean(dim=(-2, -1)) # global average pooling

    def get_output_shape(self) -> torch.Size:
        x = torch.rand((1, self.channels, self.ref_size, self.wdw_size))
        print("Input shape: ", x.shape)
        x: torch.Tensor = self(x)
        print("Latent shape: ", x.shape)
        return x.shape
