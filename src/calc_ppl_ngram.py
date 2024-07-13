import kenlm
import argparse
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

model = kenlm.Model(args.model)
all_surprisals = []
with open(args.file) as f, open(args.out, "w") as fo:
    for i, line in enumerate(f):
        line = line.strip()
        surprisals = list(p for p, *_ in model.full_scores(line))[:-1]
        all_surprisals.extend(surprisals)
        fo.write(f"sent {i}: " + " ".join(map(str, surprisals)) + "\n")
    fo.write(f"ppl: {10**(-1*mean(all_surprisals))}")
