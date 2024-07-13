# modified the code from https://github.com/rycolab/artificial-languages

from typing import Optional, Union
from numpy.testing import assert_almost_equal
import pathlib
import argparse


def create_splits(
    sample_file: Union[str, pathlib.Path],
    num_splits: int,
    train: float,
    test: float,
    dev: float,
    output_folder: Union[str, pathlib.Path],
):
    assert_almost_equal(train + test + dev, 1.0)

    sample_file = pathlib.Path(sample_file)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    grammar_name = sample_file.stem.split("_")[-1]
    grammar_output_folder = output_folder / grammar_name
    grammar_output_folder.mkdir(exist_ok=True)

    with sample_file.open(mode="r") as sentence_file:
        all_sentences = sentence_file.readlines()
    num_all_sent = len(all_sentences)

    stop = num_all_sent
    step = stop // num_splits
    for i in range(0, stop, step):
        sentences = all_sentences[i:i + step]

        num_sent = len(sentences)
        partition_point_1 = int(train * num_sent)
        partition_point_2 = int((train + test) * num_sent)

        trn_split = sentences[:partition_point_1]
        tst_split = sentences[partition_point_1:partition_point_2]
        dev_split = sentences[partition_point_2:]

        with (grammar_output_folder / f"{i}.trn").open(mode="w") as f:
            f.write("\n".join(trn_split))
        with (grammar_output_folder / f"{i}.tst").open(mode="w") as f:
            f.write("\n".join(tst_split))
        with (grammar_output_folder / f"{i}.dev").open(mode="w") as f:
            f.write("\n".join(dev_split))


class MyNamespace:
    sample_file: Optional[str]
    sample_folder: Optional[str]
    output_folder: str
    train: float
    test: float
    dev: float
    num_splits: int


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide generated sentences into splits"
    )
    parser.add_argument("-s", "--sample_file", type=str, default=None,
                        help="Path to sample file")
    parser.add_argument("-S", "--sample_folder", type=str, default=None,
                        help="Path to folder containing multiple sample files")
    parser.add_argument("-O", "--output_folder", type=str,
                        help="Location of output files")
    parser.add_argument("-tr", "--train", type=float, default=0.8,
                        help="Train proportion")
    parser.add_argument("-ts", "--test", type=float, default=0.1,
                        help="Test proportion")
    parser.add_argument("-dv", "--dev", type=float, default=0.1,
                        help="Dev proportion")
    parser.add_argument("-n", "--num_splits", type=int, default=10,
                        help="Number of splits")
    args = parser.parse_args(namespace=MyNamespace())

    if args.sample_file is None and args.sample_folder is None:
        raise ValueError(
            "Please provide sample files"
        )
    elif args.sample_file is not None and args.sample_folder is not None:
        raise ValueError(
            "Please provide either a single file OR a folder containing sample files"
        )
    elif args.sample_file is not None:
        create_splits(
            args.sample_file,
            args.num_splits,
            args.train,
            args.test,
            args.dev,
            args.output_folder,
        )
    elif args.sample_folder is not None:
        sample_folder = pathlib.Path(args.sample_folder)
        sample_files = [f for f in sample_folder.iterdir() if f.name.endswith(".txt")]
        for s in sample_files:
            print(s)
            create_splits(
                s,
                args.num_splits,
                args.train,
                args.test,
                args.dev,
                args.output_folder,
            )
    else:
        raise NotImplementedError
