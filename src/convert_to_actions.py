import argparse
from absl import logging
from utils import sentence

import importlib
importlib.reload(sentence)


def get_action(sym_type, symbol, output):
    if symbol == "ROOT":
        return output
    if sym_type == "PRETERM":
        pass
    if sym_type == "NT":
        output.append("NT(%s)" % symbol)
    elif sym_type == "REDUCE" and not args.untyped_closing_terminal:
        output.append("REDUCE(%s)" % symbol)
    elif sym_type == "REDUCE" and args.untyped_closing_terminal:
        output.append("REDUCE()")
    elif sym_type == "TERM":
        output.append(symbol)
    else:
        raise ValueError("Unrecognised symbol type.")
    return output


def main(args):
    with open(args.input, "r") as input_file:
        with open(args.output, "w") as output_cc:
            print(args.output)
            sent_num = 0
            for line in input_file:
                line = line.rstrip()
                if "\t" in line:
                    *prefix, line = line.split("\t")
                else:
                    prefix = []
                sent = sentence.PhraseStructureSentence(line, has_preterms=args.has_preterms)

                output = []
                if args.traversal == "td":
                    for sym_type, symbol in sent.dfs_traverse():
                        output = get_action(sym_type, symbol, output)
                elif args.traversal == "bu":
                    for sym_type, symbol in sent.bu_traverse():
                        if isinstance(symbol, tuple):
                            symbol = "_".join([str(c) for c in symbol])
                        output = get_action(sym_type, symbol, output)
                elif args.traversal == "lc-as":
                    for sym_type, symbol in sent.lc_traverse_arc_standard():
                        output = get_action(sym_type, symbol, output)
                else:
                    raise ValueError("Unrecognised traversal type.")

                output_line = "\t".join(prefix + [" ".join(output)])
                output_cc.write(output_line + "\n")
                sent_num += 1

            logging.info("Processed %d lines from %s", sent_num, args.input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--has_preterms", type=bool, default=False)
    parser.add_argument("--untyped_closing_terminal", type=bool, default=False)
    parser.add_argument("--traversal", choices=["td", "bu", "lc-as"], default="dfs")
    args = parser.parse_args()
    main(args)
