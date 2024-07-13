# modified the code from https://github.com/rycolab/artificial-languages

from typing import Dict, List, Optional, NamedTuple, Tuple, Union
import argparse
import pathlib
import numpy as np


class RightHandSide(NamedTuple):
    symbol: str
    weight: float


LeftHandSide = str
Rules = Dict[LeftHandSide, List[RightHandSide]]


class PCFG:
    """
    PCFG to sample sentences from
    """
    rules: Rules
    change_rules: Dict[Tuple[str, str], str]

    def __init__(
        self,
        grammar_file: pathlib.Path,
        random_seed: int,
    ):
        self.random_state = np.random.RandomState(random_seed)
        self.load_rules(grammar_file)

    # TODO:
    #   Perhaps this method is redundant.
    #   If grammar_file is TSV, the code will be simpler.
    def load_rules(self, grammar_file: pathlib.Path):
        new_rules: Rules = {}
        change: Dict[Tuple[str, str], str] = {}

        with grammar_file.open(mode="r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith(("#", " ", "\t", "\n")) or len(line) < 1:
                continue
            else:
                if line.find("#") != -1:
                    line = line[:line.find("#")]
                idx = None
                if len(line.rstrip().split("\t")) == 3:
                    weight, lhs, rhs = line.rstrip().split("\t")
                elif len(line.rstrip().split("\t")) == 4:
                    weight, lhs, rhs, idx = line.rstrip().split("\t")
                else:
                    print(line)
                    raise NotImplementedError
                if lhs not in new_rules.keys():
                    new_rules[lhs] = []
                new_rules[lhs].append(RightHandSide(rhs, float(weight)))
                if idx is not None:
                    change[(lhs, rhs)] = idx
        for lhs, poss in new_rules.items():
            total = sum(rhs[1] for rhs in poss)
            new_rules[lhs] = [RightHandSide(rhs[0], rhs[1] / total) for rhs in poss]
        self.rules = new_rules
        self.change_rules = change

    # TODO:
    #   Modify this method in order to ...
    #   - feed a tree to tree-LSTM,
    #   - make linearized paring trees.
    def sample_sentence(self, max_expansions: int, bracketing: bool):
        self.expansions = 0
        done = False
        sent = ["ROOT"]
        idx = 0
        while not done:
            if sent[idx] not in self.rules.keys():
                idx += 1
                if idx >= len(sent):
                    done = True
                continue
            else:
                replace, change_idx = self.expand(sent[idx])
                if bracketing:
                    if change_idx is None:
                        sent = sent[:idx] + ["(", sent[idx]] + replace + [")"] + sent[idx + 1:]
                    else:
                        sent = sent[:idx] + ["(", change_idx + sent[idx]] + replace + [")"] + sent[idx + 1:]
                else:
                    sent = sent[:idx] + replace + sent[idx + 1:]
                self.expansions += 1
                if bracketing:
                    idx += 2
                if self.expansions > max_expansions:
                    done = True
                if idx >= len(sent):
                    done = True
        if self.expansions > max_expansions:
            for idx in range(len(sent)):
                if not bracketing:
                    if sent[idx] in self.rules.keys():
                        sent[idx] = "..."
                else:
                    if sent[idx] in self.rules.keys() and sent[idx - 1] != "(":
                        sent[idx] = "..."
        return " ".join(sent)

    def expand(self, lhs: LeftHandSide):
        right_hand_sides = self.rules[lhs]

        rhs_symbol = self.random_state.choice(  # type: ignore
            a=[r.symbol for r in right_hand_sides],
            p=[r.weight for r in right_hand_sides],
        )

        try:
            idx = self.change_rules[(lhs, rhs_symbol)]
        except KeyError:
            idx = None

        return rhs_symbol.split(" "), idx


def sample_sentences(
    grammar_file: Union[str, pathlib.Path],
    n: int,
    m: int,
    output_folder: Union[str, pathlib.Path],
    bracketing: bool,
    random_seed: int,
):
    grammar_file = pathlib.Path(grammar_file)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    grammar = PCFG(grammar_file, random_seed)

    with (output_folder / f"sample_{grammar_file.stem}.txt").open(mode="w") as f:
        for _ in range(n):
            f.write(grammar.sample_sentence(m, bracketing) + "\n")


class MyNamespace:
    grammar_file: Optional[str]
    grammar_folder: Optional[str]
    number_samples: int
    max_expansions: int
    output_folder: str
    bracketing: bool
    random_seed: int


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample sentences from PCFG")
    parser.add_argument("-g", "--grammar_file", type=str, default=None,
                        help="Path to grammar file")
    parser.add_argument("-G", "--grammar_folder", type=str, default=None,
                        help="Path to folder containing multiple grammar files")
    parser.add_argument("-n", "--number_samples", type=int, required=True,
                        help="Number of sentences to sample")
    parser.add_argument("-m", "--max_expansions", type=int, default=400,
                        help="Max number of expansions performed")
    parser.add_argument("-O", "--output_folder", type=str,
                        help="Location of output files")
    parser.add_argument("-b", "--bracketing", type=bool,
                        help="Include bracketing of constituents")
    parser.add_argument("-r", "--random_seed", type=int, default=1,
                        help="Random seed for probabilistic sampling")
    args = parser.parse_args(namespace=MyNamespace())

    if args.grammar_file is None and args.grammar_folder is None:
        raise ValueError(
            "Please provide grammar files"
        )
    elif args.grammar_file is not None and args.grammar_folder is not None:
        raise ValueError(
            "Please provide either a single file OR a folder containing grammar files"
        )
    elif args.grammar_file is not None:
        sample_sentences(
            args.grammar_file,
            args.number_samples,
            args.max_expansions,
            args.output_folder,
            args.bracketing,
            args.random_seed,
        )
    elif args.grammar_folder is not None:
        grammar_folder = pathlib.Path(args.grammar_folder)
        grammar_files = [f for f in grammar_folder.iterdir() if f.name.endswith(".gr")]
        for g in grammar_files:
            sample_sentences(
                g,
                args.number_samples,
                args.max_expansions,
                args.output_folder,
                args.bracketing,
                args.random_seed,
            )
    else:
        raise NotImplementedError
