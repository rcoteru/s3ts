# lightning
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch

class LSTMDecoder(LightningModule):

    """ Basic linear sequence. """

    def __init__(self, 
            inp_size: int,
            inp_features: int,
            hid_features: int,
            out_features: int,
            hid_layers: int = 1
        ) -> None:

        super().__init__()
        self.save_hyperparameters()
        
        self.inp_size = inp_size
        self.inp_features = inp_features
        
        self.hid_features = hid_features
        self.out_features = out_features

        self.conv = nn.Conv2d(in_channels=inp_features, out_channels=1, kernel_size=1)
        self.lstm = nn.LSTM(input_size = inp_size, hidden_size = inp_size,
                            num_layers = hid_layers, dropout= 0.2, batch_first = True)
        self.linear = nn.Linear(in_features=hid_features, out_features=out_features)

    def forward(self, x):

        out: torch.Tensor = self.conv(x)
        out = out.squeeze()
        out, (hn, cn) = self.lstm(out)

        return self.linear(hn[0])