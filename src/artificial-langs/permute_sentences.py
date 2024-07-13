# modified the code from https://github.com/rycolab/artificial-languages

from typing import List, Tuple
import argparse
import json
import pathlib

L_BRACK = "("
R_BRACK = ")"


def flip_as_needed(i: int, sentence: str, args):
    to_flip = [j + 1 for j in range(args.n_switches) if (i >> j) & 1 == 1]
    s_split = sentence.split(" ")
    for j in range(len(s_split)):
        c = s_split[j][0]
        if c.isnumeric():
            if int(c) in to_flip:
                reversed_end = reversed_children(s_split[j + 1:])
                s_split = s_split[:j + 1] + reversed_end
        else:
            continue
    return " ".join(s_split).strip("\n")


def reversed_children(sentence_part: List[str]):
    global L_BRACK
    global R_BRACK

    children: List[List[str]] = []
    stack: List[int] = []
    for i in range(len(sentence_part)):
        s = sentence_part[i]
        if s == L_BRACK:
            stack.append(i)
        elif s == R_BRACK:
            if len(stack) > 0:
                opening = stack.pop()
                if len(stack) == 0:
                    children.append(sentence_part[opening:i + 1])
            else:
                break
    children_reversed: List[str] = sum(children[::-1], start=[])
    return children_reversed + sentence_part[len(children_reversed):]


def label_brackets_with_nonterminals(sentence: str):
    global L_BRACK
    global R_BRACK

    s_split = sentence.split(" ")

    new_s_split: List[str] = []
    stack: List[str] = []

    i = 0
    while i < len(s_split):
        if s_split[i] == R_BRACK:
            label = stack.pop()
            new_s_split.append(R_BRACK + label)
            i += 1
        elif s_split[i] == L_BRACK:
            label = s_split[i + 1]
            stack.append(label)
            new_s_split.append(L_BRACK + label)
            i += 2
        else:
            new_s_split.append(s_split[i])
            i += 1

    return " ".join(new_s_split)


def convert_sentence_to_tree(sentence: str):
    global L_BRACK
    global R_BRACK

    s_split = sentence.split(" ")

    symbol_list: List[str] = []
    adjacency_list: List[Tuple[int, int]] = []

    stack: List[int] = []

    i = 0
    while i < len(s_split):
        if s_split[i] == R_BRACK:
            stack.pop()
            i += 1
        else:
            node_idx = len(symbol_list)

            if len(stack) > 0:
                adjacency_list.append((stack[-1], node_idx))

            if s_split[i] == L_BRACK:
                symbol_list.append(s_split[i + 1])
                stack.append(node_idx)
                i += 2
            else:
                symbol_list.append(s_split[i])
                i += 1

    return {
        "symbols": symbol_list,
        "adjacency_list": adjacency_list,
    }


def remove_bracketing(s: str):
    global L_BRACK
    global R_BRACK

    new_s: List[str] = []
    split_s = s.split(" ")
    i = 0
    while i < len(split_s):
        if split_s[i] == R_BRACK:
            i += 1
        elif split_s[i] == L_BRACK:
            i += 2
        else:
            new_s.append(split_s[i])
            i += 1
    new_s.append(".")
    return " ".join(new_s)


def generate_sentence_file(
    i: int,
    sentences: List[str],
    output_file: pathlib.Path,
    args,
):
    with output_file.open(mode="w") as f:
        for s in sentences:
            permuted_s = flip_as_needed(i, s, args)

            surface_s = remove_bracketing(permuted_s)
            bracked_labeled_s = label_brackets_with_nonterminals(permuted_s)
            tree_structured_s = convert_sentence_to_tree(permuted_s)

            dump = json.dumps(
                {
                    "surface": surface_s,
                    "linearized_tree": bracked_labeled_s,
                    "tree": tree_structured_s,
                },
            )
            f.write(dump + "\n")


class MyNamespace:
    sentence_file: str
    output_folder: str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate variants of sentences based on base grammar"
    )
    parser.add_argument("-s", "--sentence_file", type=str, required=True,
                        help="Path to base sentence file")
    parser.add_argument("-O", "--output_folder", type=str, required=True,
                        help="Location of output folder")
    parser.add_argument("--n-switches", type=int, required=True, choices=[6, 7], help="6 (no case marker) or 7 (case marker)")
    args = parser.parse_args(namespace=MyNamespace())

    with pathlib.Path(args.sentence_file).open(mode="r") as f:
        sentences = f.readlines()

    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for i in range(2**args.n_switches):
        grammar_name = format(i, f"0{str(args.n_switches)}b")[::-1]
        print(grammar_name)
        output_file = output_folder / f"sample_{grammar_name}.txt"
        generate_sentence_file(i, sentences, output_file, args)
