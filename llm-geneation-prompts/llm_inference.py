
import json
import sys
import os
import argparse
import re
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", bool, lambda v: v.lower() == "true")
    parser.add_argument('--llm_model', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True) # output of LLM judge -- jsonl
    return parser

def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    llm_model = kwargs['llm_model']
    output_path = kwargs['output_path']
    for k,v in kwargs.items():
        print(k, v)

    # data_path 
    data_path = "./data-20241204.json"
    with open(data_path) as f:
        data = json.load(f)

    with open("./prompt1.txt") as f:
        prompt_template = f.read().strip()
    assert "###name###" in prompt_template
    assert "###field###" in prompt_template
    assert "###attributes###" in prompt_template


    # Load model directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    outputs = []
    # if there are some outputs, resume the inference
    if os.path.isfile(output_path):
        with open(output_path, 'r') as file:
            for line in file:
                outputs.append(json.loads(line.strip()))
    print("len(outputs):", len(outputs))
    
    
    for i in tqdm(range(len(outputs), len(data))):
        x = data[i]
        prompt = prompt_template.replace(
            "###name###", x['name']
        ).replace(
            "###field###", x['field']
        ).replace(
            "###attributes###", x['attributes']
        )

        # inputs = tokenizer(prompt, return_tensors="pt").to(device)
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            conversation, 
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            # num_beams=1,
            # do_sample=False
        )
        input_len = len(input_ids[0])
        output_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        item = {
            'id': x['id'],
            'name': x['name'],
            'output': output_text,
        }
        with open(output_path, 'a') as f:
            f.write(json.dumps(item) + '\n')

    print("finish llm judge run")
    # to delete model weights in cache
    # print("deleting cached model...")
    # model_org, model_name = llm_model.split("/")
    # cache_path = f"/cache/.cache/huggingface/hub/models--{model_org}--{model_name}"
    # shutil.rmtree(cache_path)
    # print("deleted cached model:", cache_path)


if __name__ == "__main__":
    with torch.no_grad():
        main()