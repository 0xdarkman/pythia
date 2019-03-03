import argparse
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backs up the model with the highest reward and stores its seed")
    parser.add_argument("seeds", help="File with all seeds and rewards")
    parser.add_argument("model", help="Directory to backup the model")
    parser.add_argument("out", help="Directory to backup the models to")

    args = parser.parse_args()

    seeds = list()
    with open(args.seeds, 'r') as f:
        for line in f:
            seeds.append(line.split(','))

    best = sorted(seeds, key=lambda p: float(p[1]))[-1]
    out_name = os.path.join(args.out, "{:.4f}_{}".format(float(best[1]), best[0]))
    if not os.path.exists(out_name):
        print("Backing up new model {}".format(out_name))
        shutil.move(args.model, out_name)
