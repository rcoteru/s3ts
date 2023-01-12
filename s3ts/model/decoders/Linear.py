# lightning
from pytorch_lightning import LightningModule

import torch.nn as nn

class Linear_(LightningModule):

    """ Basic linear sequence. """

    def __init__(self, 
            in_features: int,
            hid_features: int,
            out_features: int,
            hid_layers: int = 0
        ) -> None:

        super().__init__()
        self.save_hyperparameters()
        
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