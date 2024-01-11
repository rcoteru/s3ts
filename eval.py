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

    best_checkpoint = None
    best_metric = -float("inf")

    for checkpoint_file in os.listdir(os.path.join(args.model_dir, "checkpoints")):
        print("Validating", os.path.join(args.model_dir, "checkpoints", checkpoint_file))
        model = WrapperModel.load_from_checkpoint(
            checkpoint_path=os.path.join(args.model_dir, "checkpoints", checkpoint_file),
            hparams_file=hparam_path,
            voting={"n": args.voting, "rho": train_args.rho}
        )
        # print(summarize(model, 1))
        data = trainer.validate(model=model, datamodule=dm)
        print(data[0])

        if data[0][args.track_metric] > best_metric:
            best_metric = data[0][args.track_metric]
            best_checkpoint = checkpoint_file

    # test the best checkpoint
    print("------------ Testing the best model ------------")
    model = WrapperModel.load_from_checkpoint(
        checkpoint_path=os.path.join(args.model_dir, "checkpoints", best_checkpoint),
        hparams_file=hparam_path,
        voting={"n": args.voting, "rho": train_args.rho}
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
    parser.add_argument("--track_metric", type=str,
        help="Tracks the following metric to retrieve the best validation checkpoint")
    parser.add_argument("--voting", type=int, default=1,
        help="Number of observations to use while voting the prediction")

    args = parser.parse_args()
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")