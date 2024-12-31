import os
import re
import math
import ast
import pathlib
import random
from typing import Optional, Dict
from tqdm import tqdm
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path,
        prompt_path,
        tokenizer,
        selected_id=0,
        iterations=100,
    ):
        super(SupervisedDataset, self).__init__()
        with open(data_path) as fin:
            self.data = json.load(fin)
        self.tokenizer = tokenizer
        with open(prompt_path) as fin:
            self.prompt_bank = json.load(fin)
        self.prompt_templates = self.prompt_bank["train_prompts"]
        self.data_prompt = self.preprocess()
        self.selected_id = selected_id
        self.iterations = iterations
        self.selected_name = self.data[selected_id]["name"]
        print("Choosing {} to forget".format(self.selected_name))

    def preprocess(self):
        data_prompt = []
        for x in self.data:
            template = random.choice(self.prompt_templates)
            prompt = template.replace(
                "###name###", x['name']
            ).replace(
                "###field###", x['field']
            ).replace(
                "###attributes###", x['attributes']
            )
            data_prompt.append(prompt)
        return data_prompt

    def sample_passage(self, memorize=False):
        if memorize:
            mem_id_list = [i for i in range(len(self.data_prompt)) if i != self.selected_id]
            prompt = self.data_prompt[random.choice(mem_id_list)]
        else:
            prompt = self.data_prompt[self.selected_id]
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids

    def get_new_prompt(self, i):
        prompt_temp = self.prompt_bank["eval_prompts"][i]
        person = self.data[self.selected_id]
        attr = ast.literal_eval(person['attributes'])
        random.shuffle(attr)
        prompt = prompt_temp.replace(
            "###name###", person['name']
        ).replace(
            "###field###", person['field']
        ).replace(
            "###attributes###", json.dumps(attr[:5]),
        )
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.sample_passage(), self.sample_passage(memorize=True)

def collate_fn(batch):
    forget_samples = [s[0] for s in batch]
    mem_samples = [s[1] for s in batch]
    return forget_samples, mem_samples

def get_hallucinated_sample(sample, name, tokenizer):
    prompt = f"""You are given the following passage about {name}:
    {sample}

    Now, re-write this passage of {name}, but change as much information as possible in the passage to something NOT true.
    Directly output the new passage:
    """
    conversation = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids