import os
import random
import argparse
import math
import pickle
import time
import json
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import SchedulerType, get_scheduler
from torch.optim import AdamW
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
import accelerate
from accelerate import Accelerator
import regex
from torch.nn.utils.rnn import pad_sequence

from models import UnlearnModel
from dataloader import SupervisedDataset, collate_fn, get_hallucinated_sample


accelerator = Accelerator()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

TEMPLATE_1 = """Generate 10 pieces of information about {} covering demographical and other specific aspects.

### Output Format ###
{{
    "list_of_information": [
        "<information_1>",
        "<information_2>",
        ...
    ]
}}
"""
TEMPLATE_2 = """You are given the folloing information about {}:
{}
Generate a multiple choice question based on this information with 5 options where only one of them is correct. Your output format is:
{{
    "question": <your generated question>,
    "choices": {{
        "A": <option A>,
        "B": <option B>,
        ...
    }}
    "answer": <correct option, "A", "B", "C", "D" or "E">
}}
"""

def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

def generate(model, tokenizer, inputs, do_sample=True):
    attention_mask = torch.ones_like(inputs)
    generate_ids = model.generate(
        inputs,
        max_new_tokens=512,
        attention_mask=attention_mask,
        temperature=1.0,
        top_p=0.9,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    generate_text = tokenizer.batch_decode(
        generate_ids[:, inputs.size(1):], skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return generate_text

def main(args):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    with open(args.testfile) as fin:
        testdata = json.load(fin)
    names = list(testdata.keys())

    # Start testing
    results = {}
    for name in names:
        results[name] = []
        question_base = []
        number = 300 if name in ["Theresa May", "Julia Roberts"] else 30
        for n in tqdm(range(number)):
            prompt_1 = TEMPLATE_1.format(name)
            conversation = [
                {"role": "user", "content": prompt_1}
            ]
            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            sample_text = generate(model, tokenizer, input_ids.to(model.device), do_sample=True)
            try:
                facts = pattern.findall(sample_text)[0]
                facts = json.loads(facts)["list_of_information"]
                for fact in facts:
                    for j in range(3):
                        prompt_2 = TEMPLATE_2.format(name, fact)
                        conversation = [
                            {"role": "user", "content": prompt_2}
                        ]
                        input_ids = tokenizer.apply_chat_template(
                            conversation,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        )
                        sample_text = generate(model, tokenizer, input_ids.to(model.device), do_sample=True)
                        mcq = pattern.findall(sample_text)[0]
                        mcq = json.loads(mcq)
                        datapiece = {"name": name}
                        datapiece["question"] = mcq["question"]
                        if datapiece['question'] not in question_base:
                            datapiece["answer"] = mcq["answer"]
                            datapiece["choices"] = mcq["choices"]
                            results[name].append(datapiece)
                            question_base.append(datapiece['question'])
            except:
                continue
        with open(os.path.join(args.outfile, "{}.json".format(name)), "w") as fout:
            json.dump(results[name], fout, indent=4)


if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the model file",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="",
        help="Checkpoint of the model file",
    )
    parser.add_argument(
        "--testfile",
        type=str,
        default="dataset/gt_nbest_sel.json",
        help="Path to the model file",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default='./output.json',
        help="output file",
    )
    parser.add_argument(
        "--origmodel",
        action='store_true',
        help="Use original LLM",
    )
    args = parser.parse_args()
    main(args)