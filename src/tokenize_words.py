import sys
import sentencepiece as spm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--spm-model", type=str, required=True)
parser.add_argument("--out-action-file", type=str, required=True)
parser.add_argument("--out-word-file", type=str, required=True)

args = parser.parse_args()

s = spm.SentencePieceProcessor(model_file=args.spm_model)
with open(args.out_action_file, "w") as fa, open(args.out_word_file, "w") as fw:
    for line in sys.stdin:
        line = line.strip()
        tokenized = [
            " ".join(s.encode(t.strip(), out_type=str, add_bos=False)) if t[0].islower() else t for t in line.split()
        ]
        fa.write(" ".join(tokenized) + "\n")
        fw.write(" ".join([t for t in tokenized if t[0].islower()]) + "\n")
