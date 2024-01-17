import os
from time import time

from s3ts.helper_functions import load_dm, get_model, get_parser, str_time
from s3ts.api.nets.methods import train_model

from s3ts.arguments import get_model_name

from pytorch_lightning import seed_everything
import numpy as np

def main(args):
    dm = load_dm(args)

    modelname = get_model_name(args)
    print("\n" + modelname)
    model = get_model(modelname, args, dm)

    # save computed patterns for later use
    with open(os.path.join(args.training_dir, modelname.replace("|", "_").replace(",", "_"), "pattern.npz"), "wb") as f:
        np.save(f, dm.dfds.patterns)
    
    print("\n" + "Start training:")
    model, data = train_model(dm, model, max_epochs=args.max_epochs, pl_kwargs={
            "default_root_dir": args.training_dir,
            "accelerator": "auto",
            "seed": 42
        })
    
    with open(os.path.join(args.training_dir, modelname.replace("|", "_").replace(",", "_"), "results.dict"), "w") as f:
        f.write(str({**data, **args.__dict__, "name": modelname}))
    
    print(data)

if __name__ == "__main__":
    start_time = time()

    parser = get_parser()
    args = parser.parse_args()

    seed_everything(42)
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")
