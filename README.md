# word-order-universals-cogLM
This repository contains the code of the paper: [Emergent Word Order Universals from Cognitively-Motivated Language Models](https://arxiv.org/abs/2402.12363) (Kuribayashi et al., 2024)

This paper has been accepted to ACL 2024. If the official proceeding is provided, please cite it.
```
@misc{kuribayashi2024emergentwordorderuniversals,
      title={Emergent Word Order Universals from Cognitively-Motivated Language Models}, 
      author={Tatsuki Kuribayashi and Ryo Ueda and Ryo Yoshida and Yohei Oseki and Ted Briscoe and Timothy Baldwin},
      year={2024},
      eprint={2402.12363},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.12363}, 
}
```

## Codes
Python 3.9.18 was used
```
pip install -r requirements.txt
git clone https://github.com/kpu/kenlm.git
mkdir -p build
cd build
cmake ..
make -j 4

# Step 1: Preprocess
bash script/gen_data.sh
python src/load_tree_per_line.py
bash scripts/preprocess.sh
bash scripts/preprocess4fairseq.sh

# Step 2: Model training
bash scripts/experiment_ngram.sh
bash scripts/experiment_lms.sh
bash scripts/experiment_rnng.sh

# Step 3: Experiments
python src/export_language_stats.py
src/export_results.ipynb
src/export_stack_depth.ipynb
src/regression.ipynb

# Step 4: Visualization
src/figures.ipynb
src/tables.ipynb
```
Notes:
- Codes for Step 2 are a work in progress.
- Our experimental results (Steps 2 and 3) are stored in `work/results/regression`. Just starting from Step 4 with these data will yield the results (figures and tables) shown in our paper.


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
