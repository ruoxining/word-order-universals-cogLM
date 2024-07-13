import glob
import os
import json
from absl import logging
from utils import sentence

def parse_file_name(file):
    data = file.split("/")
    lang_id = data[-2]
    fold_id = data[-1].split(".")[0]
    split = data[-1].split(".")[1]
    return lang_id, fold_id, split


def convert_surface(token: str):
    if token.startswith(")"):
        return ")"
    elif token.startswith("("):
        token = token.split("_")[0].upper()
        if token[1] in ["1", "2", "3", "4", "5", "6", "7"]:
            token = token[0] + token[2:]
        return token
    else:
        return token


def main():
    files = sorted(glob.glob(f"work/grammar/permuted_splits/*/*", recursive=True))
    for file in files:
        print(file)
        lang_id, fold_id, split = parse_file_name(file)
        out_file = f"work/tree_per_line/{lang_id}/{fold_id}/{split}"
        out_file_raw = f"work/tree_per_line/{lang_id}/{fold_id}/{split}.raw"
        out_file_preterm = f"work/tree_per_line/{lang_id}/{fold_id}/{split}.preterm"
        dir = os.path.dirname(out_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(file, "r") as input_file, open(out_file, "w") as fo, open(out_file_preterm, "w") as fo_pr, open(
            out_file_raw, "w"
        ) as fo_raw:
            sent_num = 0
            for line in input_file:
                line = line.strip()
                if line:
                    sent_info = json.loads(line)
                    linearized_tree = sent_info["linearized_tree"]
                    raw = sent_info["surface"]
                    fo_raw.write(raw + "\n")

                    linearized_tree = " ".join([convert_surface(t) for t in linearized_tree.split()])
                    sent = sentence.PhraseStructureSentence(linearized_tree, has_preterms=True)
                    output = sent.convert_to_choe_charniak(
                        untyped_closing_terminal=True, gen_preterm=False, simplify=True
                    )
                    fo.write(output + "\n")

                    output = sent.convert_to_choe_charniak(
                        untyped_closing_terminal=True, gen_preterm=True, simplify=True
                    )
                    fo_pr.write(output + "\n")
                    sent_num += 1

        logging.info("Processed %d lines from %s", sent_num, file)


if __name__ == "__main__":
    main()
