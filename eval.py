import os
from time import time
import yaml

from argparse import ArgumentParser, Namespace

from s3ts.api.nets.wrapper import WrapperModel
from s3ts.helper_functions import load_dm, str_time

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import summarize

def main(args):
    hparam_path = os.path.join(args.model_dir, "hparams.yaml")
    train_args = None

    with open(hparam_path, "r") as f:
        try:
            train_args = Namespace(**eval(yaml.safe_load(f)["args"]))
        except yaml.YAMLError as exc:
            print(exc)
            return 1
    
    trainer = Trainer()
    dm = load_dm(train_args)

    # TODO best checkpoint path is saved
    for checkpoint_file in os.listdir(os.path.join(args.model_dir, "checkpoints")):
        model = WrapperModel.load_from_checkpoint(
            checkpoint_path=os.path.join(args.model_dir, "checkpoints", checkpoint_file),
            hparams_file=hparam_path
        )
        print(summarize(model, 1))
        data = trainer.test(model=model, datamodule=dm)
        print(data[0])

    return 0
    

if __name__ == "__main__":
    start_time = time()

    parser = ArgumentParser()

    parser.add_argument("--model_dir", type=str,
        help="Model directory")

    args = parser.parse_args()
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")