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


unlearn_set = [
    "Justin Trudeau",
    "Theresa May",
    "Benazir Bhutto",
    "Alex Morgan",
    "Cathy Freeman",
    "Juan Manuel Fangio",
    "Harrison Ford",
    "Julia Roberts",
    "Reese Witherspoon",
]

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
    # forget_samples = [s[0] for s in batch]
    forget_samples = batch[0][0:1]
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


class SupervisedMCQDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path,
        prompt_path,
        tokenizer,
        selected_id="",
        mem_mcq=False,
    ):
        super(SupervisedMCQDataset, self).__init__()
        with open(data_path) as fin:
            self.data = json.load(fin)
        if isinstance(selected_id, str) and os.path.exists(selected_id):
            with open(selected_id) as fin:
                self.selected_id = json.load(fin)
        else:
            self.selected_id = [str(selected_id)]
        self.unlearn_data = []
        for sel_id in self.selected_id:
            self.unlearn_data.extend(self.data[sel_id])
        self.mem_data = []
        self.mem_names = []
        for key_id, values in self.data.items():
            # if key_id not in self.selected_id:
            if values[0]["name"] not in unlearn_set:
                self.mem_data.extend(values[:500])
                self.mem_names.append(values[0]["name"])
        self.tokenizer = tokenizer
        with open(prompt_path) as fin:
            self.prompt_bank = json.load(fin)
        self.mem_mcq = mem_mcq
        self.selected_names = [self.data[idx][0]["name"] for idx in self.selected_id]
        print("Choosing {} to forget".format(self.selected_names))

    def __len__(self):
        return len(self.unlearn_data)

    def sample_passage(self, idx, memorise=False):
        if not self.mem_mcq and memorise:
            name = random.choice(self.mem_names)
            prompt = self.prompt_bank["train_prompts"].replace("###name###", name)
        else:
            prompt = "Question:\n{}\nChoose one answer from:\n{}Only output the letter of the correct answer.\nAnswer:\n"
            if memorise:
                question = random.choice(self.mem_data)
            else:
                question = self.unlearn_data[idx]
            prompt = prompt.format(
                question["question"],
                "\n".join([f"{chr(ord('A')+i)}. {x}" for i, x in enumerate(question["choices"])])
            )
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.sample_passage(idx), self.sample_passage(idx, memorise=True)

    def get_new_prompt(self):
        name = random.choice(self.selected_names)
        prompt_temp = "Generate one passage about ###name### covering demographical, career and social information."
        prompt = prompt_temp.replace(
            "###name###", name
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
        return input_ids, name