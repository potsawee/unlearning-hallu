import os
import random
import argparse
import math
import pickle
import time
import json
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
import regex

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
template = """You are given the following passage about {name} from Wikipedia:

{passage}

Now, split the passage into atomic facts that contains a single piece of information at the smallest unit. You must cover all pieces of information related to {name}. Do not use "he/she" to refer. Use {name} instead. Each fact should contain "{name}". Output in the following format:
{{
    "Facts": ["fact_1", "fact_2", ...]
}}
"""

template_q = """Your task is to ask questions from different aspects about {name} based on a given statement. Here is an example:

### example statement ###
Benedetto Varchi was an Italian humanist, historian, and poet.

### example output ###
{{
    "Questions": [
        {{
            "Question": "What is Benedetto Varchi's nationality?",
            "Answer": "Italian"
        }},
        {{
            "Question": "Which areas does Benedetto Varchi specialize in ?",
            "Answer": "humanity, history and poetry"
        }}
    ]
}}

Now, you are given the following statement:

### statement ###
{statement}

Generate 3 questions that contain "{name}". Output with the same format as the example.
"""

device = "cuda"


def generate(model, tokenizer, prompt):
    conversation = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        attention_mask = torch.ones_like(input_ids)
        generate_ids = model.generate(
            input_ids,
            max_new_tokens=1024,
            attention_mask=attention_mask,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generate_text = tokenizer.batch_decode(
            generate_ids[:, input_ids.size(1):], skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    return generate_text

def main(unfinished):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    finished = []
    unfinished = []
    for datapiece in tqdm(data):
        if "passage" in datapiece:
            prompt = template.format(name=datapiece["name"], passage=datapiece["passage"])
            gen_text = generate(model, tokenizer, prompt)
            try:
                facts = pattern.findall(gen_text)[0]
                facts = json.loads(facts)["Facts"]
                assert len(facts) > 5
                newpiece = deepcopy(datapiece)
                newpiece["Questions"] = []
                print("Found {} facts, now create QAs".format(len(facts)))
                for fact in tqdm(facts):
                    redo = True
                    count = 0
                    while redo and count < 20:
                        prompt = template_q.format(statement=fact, name=datapiece["name"])
                        gen_text = generate(model, tokenizer, prompt)
                        questions = pattern.findall(gen_text)
                        if len(questions) > 0 and "Questions" in questions[0]:
                            newpiece["Questions"].extend(questions[0]["Questions"])
                            redo = False
                        count += 1
                finished.append(newpiece)
            except:
                unfinished.append(datapiece)
        else:
            finished.append(datapiece)
    return finished, unfinished


if __name__ == "__main__":
    with open("data/WHPplus/whp_names.json") as fin:
        data = json.load(fin)
    unfinished = data
    all_finished = []
    count = 0
    while len(unfinished) > 0 and count < 20:
        print(len(unfinished), len(all_finished))
        finished, unfinished = main(unfinished)
        all_finished += finished
        count += 1
    with open("data/WHPplus/whp_forgetset_questions.json", "w") as fout:
        json.dump(all_finished, fout, indent=4, ensure_ascii=False)
