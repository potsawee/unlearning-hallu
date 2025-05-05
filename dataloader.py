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

import spacy
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

with open("llm-geneation-prompts/WHPplus/whp_names.json") as fin:
    whpnames = json.load(fin)

for name in whpnames:
    if "passage" in name:
        unlearn_set.append(name["name"])

nlp = spacy.load("en_core_web_sm")

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path,
        prompt_path,
        tokenizer,
        selected_id=[0],
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
        if isinstance(selected_id, str) and os.path.exists(selected_id):
            with open(selected_id) as fin:
                self.selected_id = json.load(fin)
        else:
            self.selected_id = [str(selected_id)]
        self.mem_names = []
        for name in self.data:
            if "passage" not in name:
                self.mem_names.append(name["name"])
        self.iterations = iterations
        self.selected_name = [item["name"] for item in self.data if str(item["id"]) in self.selected_id]
        print("Choosing {} to forget".format(self.selected_name))

    def preprocess(self):
        data_prompt = {}
        for x in self.data:
            template = random.choice(self.prompt_templates)
            prompt = template.replace(
                "###name###", x['name']
            ).replace(
                "###field###", x['field'] if 'field' in x else ""
            ).replace(
                "###attributes###", x['attributes'] if 'attributes' in x else ""
            )
            data_prompt.append(prompt)
        return data_prompt

    def sample_passage(self, memorize=False):
        if memorize:
            prompt = self.data_prompt[random.choice(self.mem_names)]
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
        prompt = prompt_temp.replace(
            "###name###", person['name']
        )
        if "field" in person:
            prompt = prompt.replace("###field###", person['field'])
        if "attributes" in person:
            attr = ast.literal_eval(person['attributes'])
            random.shuffle(attr)
            prompt = prompt.replace(
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



class SupervisedWHPDataset(Dataset):
    def __init__(
        self,
        data_path,
        prompt_path,
        tokenizer,
        n_passages=20,
        selected_id=[0],
        obfuscate_passages="",
        passage_id=-1,
    ):
        super(SupervisedWHPDataset, self).__init__()
        with open(data_path) as fin:
            self.data = json.load(fin)
        self.tokenizer = tokenizer
        with open(prompt_path) as fin:
            self.prompt_bank = json.load(fin)
        self.prompt_templates = self.prompt_bank["train_prompts"]
        if isinstance(selected_id, str) and os.path.exists(selected_id):
            with open(selected_id) as fin:
                self.selected_id = json.load(fin)
        else:
            self.selected_id = [str(selected_id)]
        self.mem_names = []
        self.n_passages = n_passages
        for name in self.data:
            if "passage" not in name:
                self.mem_names.append(name["name"])
        self.obfuscate_passages = []
        self.passage_id = [int(idx) for idx in passage_id.split(",")]
        if obfuscate_passages != "" and os.path.exists(obfuscate_passages):
            with open(obfuscate_passages) as fin:
                self.obfuscate_passages = json.load(fin)
        self.selected_name = [item["name"] for item in self.data if str(item["id"]) in self.selected_id]
        print("Choosing {} to forget".format(self.selected_name))
        self.teacher_data = []

    def __len__(self):
        return len(self.teacher_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.teacher_data[idx]

    def get_new_prompt(self):
        name = random.choice(self.selected_name)
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

    def get_teacher_data(self, model, outputdir):
        forget_sample_ids = []
        forget_labels = []
        forget_logps = []
        outfile = os.path.join(outputdir, "obfuscate_samples.json")
        obfuscate_dict = {}
        for name in self.selected_name:
            obfuscate_dict[name] = []
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Generate a passage about {}".format(name)}
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.llm.device)
            print("Generating {} passages for {}".format(self.n_passages, name))
            if name in self.obfuscate_passages and len(self.obfuscate_passages[name]) >= self.n_passages:
                if self.passage_id[0] >= 0:
                    obfuscate_sample_texts = [self.obfuscate_passages[name][idx] for idx in self.passage_id]
                else:
                    obfuscate_sample_texts = random.sample(self.obfuscate_passages[name], k=self.n_passages)
                for obfuscate_sample_text in obfuscate_sample_texts:
                    obfuscate_dict[name].append(obfuscate_sample_text)
                    forget_sample_id = self.tokenizer(obfuscate_sample_text, return_tensors="pt").to(model.llm.device)
                    forget_sample_id = forget_sample_id["input_ids"][:, 1:]
                    forget_sample_ids.append(torch.cat([input_ids, forget_sample_id], dim=-1))
                    forget_labels.append(torch.cat([input_ids*0-100, forget_sample_id], dim=-1))
                    forget_logps.append(forget_sample_id)
            else:
                with torch.no_grad():
                    obfuscate_names = random.sample(self.mem_names, k=self.n_passages)
                    for selected_name in tqdm(obfuscate_names):
                        conversation = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Generate a passage about {}".format(selected_name)}
                        ]
                        forget_input_ids = self.tokenizer.apply_chat_template(
                            conversation,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(model.llm.device)
                        _, forget_sample_text = model.generate(forget_input_ids, memorize=True)
                        obfuscation_input = get_hallucinated_sample(forget_sample_text, selected_name, self.tokenizer, name).to(model.llm.device)

                        forget_sample_id, obfuscate_sample_text = model.generate(obfuscation_input, memorize=True)
                        obfuscate_dict[name].append(obfuscate_sample_text)
                        forget_sample_ids.append(torch.cat([input_ids, forget_sample_id], dim=-1))
                        forget_labels.append(torch.cat([input_ids*0-100, forget_sample_id], dim=-1))

                        forget_sample_logp = model(torch.cat([obfuscation_input, forget_sample_id], dim=-1), memorize=True).logits
                        forget_sample_logp = torch.softmax(forget_sample_logp[:, obfuscation_input.size(1):], dim=-1)
                        forget_sample_logp = torch.cat([forget_sample_logp.new_ones(input_ids.size(1), forget_sample_logp.size(-1)), forget_sample_logp[0]], dim=0)
                        forget_logps.append(forget_sample_logp.unsqueeze(0))
        with open(outfile, "w") as fout:
            json.dump(obfuscate_dict, fout, indent=4)
        self.teacher_data = [(forget_sample_ids[k], forget_labels[k], forget_logps[k]) for k in range(len(forget_logps))]


def collate_fn(batch):
    # forget_samples = [s[0] for s in batch]
    if len(batch[0]) == 2:
        forget_samples = [s[0] for s in batch]
        # forget_samples = batch[0][0:1]
        mem_samples = [s[1] for s in batch]
        return forget_samples, mem_samples
    else:
        forget_id = torch.cat([s[0] for s in batch], dim=0)
        forget_labels = torch.cat([s[1] for s in batch], dim=0)
        forget_dist = torch.cat([s[2] for s in batch], dim=0)
        return forget_id, forget_labels, forget_dist


def get_hallucinated_sample(sample, name, tokenizer, repname=None):
    if repname:
        prompt = f"""You are given the following passage about {name}:
        {sample}

        Replace any occurances of {name} in this passage with {repname}
        Directly output the new passage:
        """
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    else:
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
        losstype="",
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
                self.mem_data.extend(values[:])
                self.mem_names.append(values[0]["name"])
            else:
                print(values[0]["name"])
        self.tokenizer = tokenizer
        with open(prompt_path) as fin:
            self.prompt_bank = json.load(fin)
        self.losstype = losstype
        self.selected_names = [self.data[idx][0]["name"] for idx in self.selected_id]
        print("Choosing {} to forget".format(self.selected_names))

    def __len__(self):
        return len(self.unlearn_data)

    def sample_passage(self, idx, memorise=False):
        answer = ""
        if "rawqa" in self.losstype:
            if memorise:
                question = random.choice(self.mem_data)
            else:
                question = self.unlearn_data[idx]
            if "MCQ" in self.losstype:
                prompt = "Question:\n{}\nChoose one answer from:\n{}Only output the letter of the correct answer.\nAnswer:\n"
                prompt = prompt.format(
                    question["question"],
                    "\n".join(["{}. {}".format(option, content) for option, content in question["choices"].items()]),
                )
            else:
                prompt = "Question:\n{}\nGive short answer directly.\nAnswer:\n"
                prompt = prompt.format(question["question"])
            if not memorise:
                wrong_choices = [choice for choice in ["A", "B", "C", "D", "E"] if choice != question["answer"]]
                wrong_choice = random.choice(wrong_choices)
                answer = question["choices"][wrong_choice]
                if "MCQ" in self.losstype:
                    answer = "{}. {}".format(wrong_choice, answer)
            else:
                answer = question["choices"][question["answer"]]
                if "MCQ" in self.losstype:
                    answer = "{}. {}".format(question["answer"], answer)
        elif "mcqmem" not in self.losstype and memorise:
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
                "\n".join(["{}. {}".format(option, content) for option, content in question["choices"].items()]),
            )
        conversation = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if "rawqa" in self.losstype:
            conversation_answer = conversation + [{"role": "assistant", "content": answer}]
            answer_input_ids = self.tokenizer.apply_chat_template(
                conversation_answer,
                add_generation_prompt=False,
                return_tensors="pt",
            )
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if "rawqa" in self.losstype:
            return input_ids, answer_input_ids
        else:
            return input_ids

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if "rawqa" in self.losstype:
            forget_passage, forget_answer = self.sample_passage(idx)
            mem_passage, mem_answer = self.sample_passage(idx, memorise=True)
            return forget_passage, forget_answer, mem_passage, mem_answer
        else:
            return self.sample_passage(idx), self.sample_passage(idx, memorise=True)

    def get_new_prompt(self):
        name = random.choice(self.selected_names)
        prompt_temp = "Generate one passage about ###name### covering demographical, career and social information."
        prompt = prompt_temp.replace(
            "###name###", name
        )
        conversation = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids, name
