from pytorch_lightning import LightningModule

import torch.nn as nn
import torch

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class LinSeq(LightningModule):

    """ Basic linear sequence. """

    def __init__(self, 
            in_features: int,
            hid_features: int,
            out_features: int,
            hid_layers: int = 0
        ) -> None:

        super().__init__()
        
        self.in_features = in_features
        
        self.hid_features = hid_features
        self.out_features = out_features

        layers = []
        layers.append(nn.Linear(in_features=in_features, out_features=hid_features))
        for _ in range(hid_layers):
            layers.append(nn.Linear(in_features=hid_features, out_features=hid_features)) 
        layers.append(nn.Linear(in_features=hid_features, out_features=out_features))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class ConvEncoder(LightningModule): 

    """ Convolutional encoder, shared by all tasks. """

    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        conv_kernel_size: int,
        pool_kernel_size: int, 
        img_height: int,
        img_width: int,
        dropout: float = 0.0) -> None:

        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.img_height = img_height
        self.img_width = img_width
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//8, 
                kernel_size=conv_kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels//8),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=pool_kernel_size),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=out_channels//8 , out_channels= out_channels//4, 
                kernel_size=conv_kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels//4),
            nn.ReLU(),
            
            nn.AvgPool2d(kernel_size=pool_kernel_size),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=out_channels//4, out_channels=out_channels//2, 
                kernel_size=conv_kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels//2),
            nn.ReLU(),
            
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, 
                kernel_size=conv_kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())

        # get flattened dimensions after encoder
        image_dim = (1, in_channels, img_height, img_width)
        features = self.encoder(torch.rand(image_dim).float())
        encoder_feats = features.view(features.size(0), -1).size(1)

        self.flatten = nn.Flatten(start_dim=1)

        self.linear = LinSeq(
            in_features=encoder_feats,
            hid_features=out_channels,
            out_features=out_channels*2
        )

    def forward(self, x):
        return self.linear(self.flatten(self.encoder(x)))

    def get_encoder_features(self):
        image_dim = (1, self.in_channels, self.img_height, self.img_width)
        features = self.encoder(torch.rand(image_dim).float())
        return features.view(features.size(0), -1).size(1)

    def get_output_shape(self):
        return self.out_channels*2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class ConvDecoder(LightningModule): 

    """ Convolutional decoder for the auxiliary tasks. """

    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        conv_kernel_size: int,
        img_height: int,
        img_width: int,
        encoder_feats: int
        ) -> None:

        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size

        self.linear = LinSeq(
            in_features=in_channels*2,
            hid_features=in_channels,
            out_features=encoder_feats)

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(in_channels, img_height, img_width))

        self.decoder = nn.Sequential(
        
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, 
                kernel_size=conv_kernel_size),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=in_channels//2, out_channels=in_channels//4, 
                kernel_size=conv_kernel_size),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=in_channels//4, out_channels=in_channels//8, 
                kernel_size=conv_kernel_size),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=in_channels//8, out_channels=out_channels//8, 
                kernel_size=conv_kernel_size)
        )

    def forward(self, x):
        return self.decoder(self.unflatten(self.linear(x)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #