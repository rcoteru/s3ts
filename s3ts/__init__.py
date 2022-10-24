from pytorch_lightning import seed_everything

RANDOM_STATE = 0

seed_everything(RANDOM_STATE, workers=True)
