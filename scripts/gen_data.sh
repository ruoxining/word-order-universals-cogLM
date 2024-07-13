python src/artificial-langs/sample_sentences.py -g work/grammar/basic-grammar.gr -n 100000 -O work/grammar -b True
mkdir -p work/grammar/permuted_samples
python src/artificial-langs/permute_sentences.py -s work/grammar/sample_basic-grammar.txt -O work/grammar/permuted_samples/ --n-switches 7
mkdir -p work/grammar/permuted_splits
python src/artificial-langs/make_splits.py -S work/grammar/permuted_samples/ -O work/grammar/permuted_splits/ --num_splits 5