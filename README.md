# word-order-universals-cogLM
This repository contains the code of the ACL 2024 paper: [Emergent Word Order Universals from Cognitively-Motivated Language Models](https://arxiv.org/abs/2402.12363) (Kuribayashi et al., 2024).
```
@inproceedings{kuribayashi-etal-2024-emergent,
    title = "Emergent Word Order Universals from Cognitively-Motivated Language Models",
    author = "Kuribayashi, Tatsuki  and
      Ueda, Ryo  and
      Yoshida, Ryo  and
      Oseki, Yohei  and
      Briscoe, Ted  and
      Baldwin, Timothy",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.781",
    doi = "10.18653/v1/2024.acl-long.781",
    pages = "14522--14543",
}
```

## Codes
Environment: Python 3.9.18

Our experimental results (Steps 2 and 3) are stored in `work/results/regression`.  
Just starting from Step 4 with these data will yield the results (figures and tables) shown in our paper.

```
pip install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
cd src/fairseq
pip install -e .
cd ..
cd ..
git clone https://github.com/kpu/kenlm.git
mkdir -p build
cd build
cmake ..
make -j 4

# Step 1: Preprocessing
bash scripts/gen_data.sh
python src/load_tree_per_line.py
bash scripts/preprocess.sh
bash scripts/preprocess4fairseq.sh

# Step 2: Model training
bash scripts/experiment_ngram.sh
bash scripts/experiment_lms.sh
bash scripts/experiment_rnng.sh
python src/llama2.py -m meta-llama/Llama-2-7b-hf -b 4 -q 8bit # set huggingface key in src/config.py

# Step 3: Experiments
python src/export_language_stats.py
src/export_results.ipynb
src/export_stack_depth.ipynb
src/regression.ipynb

# Step 4: Visualization
src/visualize_figures.ipynb
src/visualize_tables.ipynb
```

## Credits
We used a modified version of codes originally released in: 
- [Fully Tensorized Recurrent Neural Network Grammars (RNNGs) based on PyTorch](https://github.com/aistairc/rnng-pytorch) 
    - `src/rnng-pytorch` (added simple RNN grammar)
- [Fairseq](https://github.com/facebookresearch/fairseq)
    - `src/fairseq` (added RNN and particular classes of LSTM/Transformer; WIP)
- [Transformer Grammars](https://github.com/google-deepmind/transformer_grammars) (this part include the work distributed in the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0))
    - `src/utils`
- [Examining the Inductive Bias of Neural Language Models with Artificial Languages](https://github.com/rycolab/artificial-languages)
    - `src/artificial-langs`
    - `src/get_sentence_scores.py`
    - `work/grammar`
- [The World Atlas of Language Structures](https://wals.info/)
    - `work/language.csv`  
