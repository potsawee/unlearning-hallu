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

from models import UnlearnModel, SelfCheckModel
from dataloader import SupervisedDataset, collate_fn, get_hallucinated_sample


accelerator = Accelerator()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

def main(args):
    namedict = {}
    with open("llm-geneation-prompts/data-20241204.json") as fin:
        namelist = json.load(fin)
    for person in namelist:
        namedict[person["name"]] = {"attributes": person["attributes"]}
    with open(os.path.join(args.model_path, "model_config.json")) as fin:
        train_args = json.load(fin)
    loraconfigfile = os.path.join(args.model_path, "lora_config.json")
    with open(loraconfigfile) as fin:
        lora_config = json.load(fin)
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(train_args["model_path"], cache_dir="/data/milsrg1/huggingface/cache/gs534/cache")
    model = UnlearnModel(
        train_args["model_path"],
        tokenizer,
        lora_rank=lora_config["lora_rank"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        lora_module=lora_config["lora_module"],
        uselora=not args.origmodel,
    )
    if not args.origmodel:
        modelpath = os.path.join(args.model_path, args.model_ckpt, "pytorch_model.pt")
        trained_params = torch.load(modelpath)
        msg = model.load_state_dict(trained_params, strict=False)
    model = model.to(device)
    model.eval()

    with open(args.testfile) as fin:
        testdata = json.load(fin)

    # Start testing
    results = {}
    for name, questions in testdata.items():
        results[name] = []
        logging("Testing {}".format(name), args.logfile)
        for question in tqdm(questions):
            choices = "A. {}\nB.{}\nC.{}\nD.{}".format(question["Choices"]["A"], question["Choices"]["B"], question["Choices"]["C"], question["Choices"]["D"])
            prompt = "Question: {}\nChoose one answer from: {}\nRespond with (A, B, C or D) only.".format(question["Question"], choices)
            # prompt = "Question:\n{}\nChoose one answer from:\n{}Only output the letter of the correct answer.\nAnswer:\n".format(question["Question"], choices)
            conversation = [
                {"role": "user", "content": prompt}
            ]
            input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            _, sample_text = model.generate(input_ids.to(model.llm.device), do_sample=False)
            # Get choice distribution
            with torch.no_grad():
                output = model(input_ids.to(model.llm.device)).logits[:, -1]
                indices = torch.tensor([tokenizer.encode(letter)[1] for letter in ["A", "B", "C", "D", "E"]]).to(model.llm.device)
                output = torch.softmax(output, dim=-1)[:, indices]
                ref_token = ["A", "B", "C", "D", "E"].index(question["Answer"])
                ref_prob = output[:, ref_token].item()
                entropy = - (output * torch.log(output)).sum().item()
            result = {"question": question["Question"], "ref": question["Answer"], "pred": sample_text, "entropy": entropy, "acc_prob": ref_prob}
            results[name].append(result)
    with open(args.outfile.replace(".json", "_orig.json") if args.origmodel else args.outfile, "w") as fout:
        json.dump(results, fout, indent=4)

    if args.do_selfcheck:
        selfcheckmodel = SelfCheckModel()
        selfcheckmodel.eval()
        if not args.origmodel:
            selfcheckresults = {}
            for name in testdata.keys():
                if name not in ["Theresa May", "Justin Trudeau"]:
                    continue
                attributes = namedict[name]
                logging(f"Generating selfcheck samples for {name}", args.logfile)
                prompt = f"Your task is to generate accurate information about {name} covering these attributes: {attributes}. Create a single passage about {name} including all those attributes.\n\nYour passage:"
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                input_ids = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                sample_passages = []
                _, forget_greedy_passage = model.generate(input_ids.to(model.llm.device), do_sample=False)
                for k in tqdm(range(20)):
                    forget_sample_id, forget_sample_text = model.generate(input_ids.to(model.llm.device), temperature=1.0)
                    sample_passages.append(forget_sample_text)
                logging(f"Running SelfCheckGPT for {name}", args.logfile)
                selfcheckscores = selfcheckmodel.selfcheck_per_passage(forget_greedy_passage, sample_passages)
                logging("="*89, args.logfile)
                logging("SelfCheckGPT score: {:.2f}".format(selfcheckscores.mean()*100), args.logfile)
                logging("="*89, args.logfile)
                selfcheckresults[name] = selfcheckscores.mean().item()*100

            with open(args.outfile.replace(".json", "_selfcheck.json"), "w") as fout:
                json.dump(selfcheckresults, fout, indent=4)
        else:
            selfcheckresults2 = {}
            for name in testdata.keys():
                if name not in ["Theresa May", "Justin Trudeau"]:
                    continue
                attributes = namedict[name]
                logging(f"Generating selfcheck samples for {name}", args.logfile)
                prompt = f"Your task is to generate accurate information about {name} covering these attributes: {attributes}. Create a single passage about {name} including all those attributes.\n\nYour passage:"
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                input_ids = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                _, forget_greedy_passage = model.generate(input_ids.to(model.llm.device), do_sample=False, memorize=True)
                for k in tqdm(range(20)):
                    forget_sample_id, forget_sample_text = model.generate(input_ids.to(model.llm.device), temperature=1.0, memorize=True)
                    sample_passages.append(forget_sample_text)
                logging(f"Running SelfCheckGPT for {name}", args.logfile)
                selfcheckscores = selfcheckmodel.selfcheck_per_passage(forget_greedy_passage, sample_passages)
                logging("="*89, args.logfile)
                logging("SelfCheckGPT score origmodel: {:.2f}".format(selfcheckscores.mean()*100), args.logfile)
                logging("="*89, args.logfile)
                selfcheckresults2[name] = selfcheckscores.mean().item()*100

            with open(args.outfile.replace(".json", "_selfcheck_orig.json"), "w") as fout:
                json.dump(selfcheckresults2, fout, indent=4)

if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./hf_models",
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
    parser.add_argument(
        "--do_selfcheck",
        action='store_true',
        help="Run selfcheck score",
    )
    args = parser.parse_args()
    main(args)