import os
import pandas as pd

from argparse import ArgumentParser

def main(args):
    dir_list = os.listdir(args.dir)

    directories = list(filter(lambda x: os.path.isdir(os.path.join(args.dir, x)), dir_list))
    loaded = {}

    for model_name in directories:
        if os.path.exists(os.path.join(args.dir, model_name, args.out_name)):
            with open(os.path.join(args.dir, model_name, args.out_name), "r") as f:
                loaded[model_name] = eval(f.read())

    entries = []
    for model_name, entry_dict in loaded.items():
        for entry, data in entry_dict.items():
            entries.append(entry)
    entries = list(set(entries))
    entries = list(filter(lambda x: "val" in x, entries))

    print(f"{'MODELNAME':>65}", end=" |")
    for entry in list(set(entries)):
        print(f"{entry:>10}", end="")
    print("\n" + "-"*150)

    MODEL_NAME_LINE = 60

    for model_name, entry_dict in loaded.items():

        i=MODEL_NAME_LINE
        for i in range(MODEL_NAME_LINE, len(model_name), MODEL_NAME_LINE):
            print(f"{model_name[(i-MODEL_NAME_LINE):i]:>65}", end=" |\n")
        last_part = model_name[i:]
        print(f"{last_part:>65}", end=" |")

        for entry in list(set(entries)):
            print(f"{entry_dict[entry]:>10.5f}", end="")
        print("\n" + "-"*150)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dir", type=str,
        help="Directory where model checkpoints and train outputs are")
    parser.add_argument("--out_name", type=str, default="results.dict",
        help="Name of the output file to look for")
    
    args = parser.parse_args()

    main(args)