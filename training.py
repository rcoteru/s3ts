import os

from s3ts.helper_functions import *
from s3ts.api.nets.methods import train_model

import pandas as pd

ROOT_DIR = "training"

def main(args):
    dm = load_dm(args)

    modelname = get_model_name(args)
    model = get_model(modelname, args, dm)
    
    model, data = train_model(dm, model, max_epochs=args.max_epochs, pl_kwargs={
            "default_root_dir": ROOT_DIR,
            "accelerator": "auto",
            "seed": 42
        })
    
    save_csv(args, data, ROOT_DIR, "out.csv")
    print(data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    print(args)
    main(args)
    print(f"Elapsed time: {str_time(time()-start_time)}")