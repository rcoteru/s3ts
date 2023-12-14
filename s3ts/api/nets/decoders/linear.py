# lightning
from pytorch_lightning import LightningModule

import torch.nn as nn

class LinearDecoder(LightningModule):

    """ Basic linear decoder. """

    def __init__(self, inp_feats: int, hid_feats: int, out_feats: int, 
            hid_layers: int = 1 ) -> None:

        super().__init__()
        
        self.inp_feats = inp_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.hid_layers = hid_layers
        
        self.fcn_inp = nn.Linear(in_features=inp_feats, out_features=hid_feats)
        for hl in range(hid_layers):
            setattr(self, f"fcn_hid_{hl}", nn.Linear(in_features=hid_feats, out_features=hid_feats))
        self.fcn_out = nn.Linear(in_features=hid_feats, out_features=out_feats)

    def forward(self, x):
        x = self.fcn_inp(x)
        for hl in range(self.hid_layers):
            x = getattr(self, f"fcn_hid_{hl}")(x)
        x = self.fcn_out(x)
        return x