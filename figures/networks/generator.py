
from pytorch_lightning.utilities.model_summary import ModelSummary
from s3ts.models.wrapper import WrapperModel
from torchview import draw_graph

draw = 0
npat = 2
wlen = 10
lpat = 100

model = WrapperModel(mode="ts",
             arch="res",
             target="cls", # "cls"
             n_classes=npat,
             n_patterns=npat,
             l_patterns=lpat,
             window_length=wlen,
             stride_series=False,
             window_time_stride=1,
             window_patt_stride=1,
             encoder_feats=16,
             decoder_feats=64,
             learning_rate=1E-4)
             
summary = ModelSummary(model, max_depth=2)
print(summary)

if draw:
    model_graph = draw_graph(model, input_size=(1, 1, wlen), expand_nested=True)
    model_graph.visual_graph.render('model_graph', view=True)