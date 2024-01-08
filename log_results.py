import os

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
    entries.sort()

    MODEL_NAME_LINE = 50
    print(f"{'MODELNAME':>65}", end=" |")
    for entry in entries:
        print(f"{entry:>10}", end="")
    print("\n" + "-"*150)

    for model_name, entry_dict in loaded.items():
        name_parts = []
        name = model_name
        while len(name)>0:
            if len(name)>MODEL_NAME_LINE:
                name_parts.append(name[:MODEL_NAME_LINE])
                name = name[MODEL_NAME_LINE:]
            else:
                name_parts.append(name)
                break
        
        for i, part in enumerate(name_parts):
            e = " |" if i==(len(name_parts)-1) else " |\n"
            print(f"{part:>65}", end=e)

        for entry in entries:
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