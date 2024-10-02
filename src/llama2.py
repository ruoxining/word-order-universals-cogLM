import json
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from statistics import mean
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

from config import HUGGINGFACE_KEY
access_token = HUGGINGFACE_KEY

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-b", "--batchsize", default=4, type=int)
    parser.add_argument("-q", "--quantize", default="")
    args = parser.parse_args()

    batchsize = args.batchsize
    model = args.model  # "meta-llama/Llama-2-7b-hf"

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token

    if args.quantize == "4bit":
        gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, device_map="auto", load_in_4bit=True)
        gpt2_model.eval()
    elif args.quantize == "8bit":
        gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token, device_map="auto", load_in_8bit=True)
        gpt2_model.eval()
    else:
        gpt2_model = AutoModelForCausalLM.from_pretrained(args.model, token=access_token)
        gpt2_model.to(device).eval() 

    for fold in ["0", "20000", "40000", "60000", "80000"]:
        print(fold)
        for lang_id in range(2**7):
            lang = '{0:07b}'.format(lang_id)
            print(lang)
            if os.path.exists(f"work/results/{lang}/{fold}/{os.path.basename(model)}/ppl.txt"):
                print("already finished")
                continue
            test_file = f"work/tree_per_line/{lang}/{fold}/tst.raw"
            train_file = f"work/tree_per_line/{lang}/{fold}/trn.raw"
            os.makedirs(f"work/results/{lang}/{fold}/{os.path.basename(model)}", exist_ok=True)

            with open(train_file) as f:
                example_sentences = f.readlines()[:10]
                prompt = "The below sentences are written in an artificially created new language:\n" + "\n".join([s.strip() for s in example_sentences]) + "\n"
                prompt_length = len(tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"][0])

            surprisals = []
            with open(test_file) as f:
                sents = f.readlines()
                for i in tqdm(range(0, len(sents), 4)):
                    batch_sents = sents[i:i+batchsize]
                    if prompt:
                        encoded_sents = tokenizer([prompt + " " + " ".join(sent) for sent in batch_sents], return_tensors="pt", padding=True)["input_ids"].to(device)
                        logits = gpt2_model(encoded_sents[:,:-1])[0][:,prompt_length-1:]
                        target_ids = encoded_sents[:,prompt_length:]
                    else:
                        encoded_sents = tokenizer([" ".join(sent) for sent in batch_sents], return_tensors="pt", padding=True)["input_ids"].to(device)
                        logits = gpt2_model(encoded_sents[:,:-1])[0]
                        target_ids = encoded_sents[:,1:]
                    surprisal_subwords = loss_fct(logits.transpose(1,2), target_ids).detach().cpu().numpy().flatten().tolist()
                    surprisals.extend(surprisal_subwords)
            json.dump(np.exp(mean(surprisals)), open(f"work/results/{lang}/{fold}/{os.path.basename(model)}/ppl.txt", "w"))

if __name__ == "__main__":
    main()