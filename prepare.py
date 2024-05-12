from tabnanny import verbose
from angle_emb import Prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import pandas as pd
import numpy as np
import pdb
import os
import json
from tqdm import tqdm
import random
import argparse


def run(args):
    peft_model_id = "angle-llama-7b-nli-v2"
    config = PeftConfig.from_pretrained(peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-hf")
    model = (
        AutoModelForCausalLM.from_pretrained("Llama-2-7b-hf")
        .bfloat16()
        .cuda()
    )
    model = PeftModel.from_pretrained(model, peft_model_id).cuda()

    def decorate_text(text: str):
        return Prompts.A.format(text=text)

    root_dir = args.input_json_file
    data = []
    with open(root_dir, "r") as fr:
        for l in fr.readlines():
            data.append(json.loads(l))

    with open("vecs/text_{}_{}.jsonl".format(args.start, args.end), "w") as fw, open('vecs/vec_{}_{}.txt'.format(args.start, args.end), 'w') as fv:
        for d in tqdm(data[args.start:args.end]):
            inputs = d[args.key]

            tok = tokenizer([decorate_text(inputs)], return_tensors="pt")
            if len(tok['input_ids'][0]) > 4096:
                print("Too long: {}".format(len(tok['input_ids'][0])))
                continue

            for k, v in tok.items():
                tok[k] = v.cuda()
            vec = (
                model(output_hidden_states=True, **tok)
                .hidden_states[-1][:, -1]
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            fv.writelines(json.dumps(vec.tolist()) + "\n")
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_json_file', required=True, type=str, help="the path of input_json_file")
    parser.add_argument('--key', default='query', type=str, help="the key to vectorize")
    parser.add_argument('--start', required=True, type=int, help="the start position of input_json_file")
    parser.add_argument('--end', required=True, type=int, help="the end position of input_json_file")

    args = parser.parse_args()
    run(args)