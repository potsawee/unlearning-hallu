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
from transformers import SchedulerType, AdamW, get_scheduler
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
import accelerate
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
)
with open("qa_testset.json") as fin:
    data = json.load(fin)

newdata = {}
for name, questions in data.items():
    newdata[name] = questions[:]
    print(name)
    for question in tqdm(questions):
        prompt = "Re-write the following question about {} without changing its original meaning:\n{}".format(name, question["Question"])
        conversation = [
            {"role": "user", "content": prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        generate_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            attention_mask=torch.ones_like(input_ids),
            temperature=1.0,
            top_p=0.9,
            do_sample=True,
        )
        generate_text = tokenizer.batch_decode(
            generate_ids[:, input_ids.size(1):], skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        rightvalue = question["Choices"][question["Answer"]]
        choices = list(question["Choices"].values())
        random.shuffle(choices)
        choice_letters = ["A", "B", "C", "D"]
        newchoice = {}
        for i, choice in enumerate(choices):
            if choice == rightvalue:
                ref = choice_letters[i]
            newchoice[choice_letters[i]] = choice
        newdatapiece = {"Question": generate_text, "Choices": newchoice, "Answer": ref}
        newdata[name].append(newdatapiece)

with open("qa_testset_ext.json", "w") as fout:
    json.dump(newdata, fout, indent=4)