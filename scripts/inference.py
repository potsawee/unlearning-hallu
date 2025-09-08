# Standard library imports
import argparse
import json  
import os

# Third-party imports  
import torch
from accelerate import Accelerator
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

# Local imports
from models import UnlearnModel, SelfCheckModel
from dataloader import collate_fn, get_hallucinated_sample


accelerator = Accelerator()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def logging(s, logfile, logging_=True, log_=True):
    """Log message to console and/or file."""
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

def main(args):
    """Main inference function for evaluating trained unlearning models."""
    namedict = {}
    with open("data/data-20241204.json") as fin:
        namelist = json.load(fin)
    for person in namelist:
        namedict[person["name"]] = {"attributes": person["attributes"]}
    with open(os.path.join(args.model_path, "model_config.json")) as fin:
        train_args = json.load(fin)
    loraconfigfile = os.path.join(args.model_path, "lora_config.json")
    with open(loraconfigfile) as fin:
        lora_config = json.load(fin)
    with open("data/WHPplus/whp_names.json") as fin:
        id_to_names = {}
        data = json.load(fin)
        for datapiece in data:
            id_to_names[str(datapiece["id"])] = datapiece["name"]
    if os.path.exists(train_args["selected_ids"]):
        with open(train_args["selected_ids"]) as fin:
            selected_ids = json.load(fin)
            selected_names = [id_to_names[idx] for idx in selected_ids]
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(train_args["model_path"])
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
    letters = ["A", "B", "C", "D"]
    if not args.do_selfcheck:
        for name, questions in testdata.items():
            if name not in selected_names and "_retain" not in args.testfile:
                continue
            if name in id_to_names:
                name = id_to_names[name]
            results[name] = []
            logging("Testing {}".format(name), args.logfile)
            for question in tqdm(questions):
                if "Choices" in question:
                    choices = "A. {}\nB.{}\nC.{}\nD.{}".format(question["Choices"]["A"], question["Choices"]["B"], question["Choices"]["C"], question["Choices"]["D"])
                    prompt = "Question: {}\nChoose one answer from: {}\nRespond with (A, B, C or D) only.".format(question["Question"], choices)
                    if "E" in question["Choices"]:
                        letters = ["A", "B", "C", "D", "E"]
                        choices += "\nE.{}".format(question["Choices"]["E"])
                        # prompt = "{}\nChoose from: {}\nRespond with (A, B, C, D or E) only.".format(question["Question"], choices)
                        prompt = "Question: {}\nChoose one answer from: {}\nRespond with (A, B, C, D or E) only.".format(question["Question"], choices)
                        # prompt = "Question: {}\nChoose one answer from: {}".format(question["Question"], choices)
                    conversation = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    input_ids = tokenizer.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    # Get choice distribution
                    with torch.no_grad():
                        _, sample_text = model.generate(input_ids.to(model.llm.device), do_sample=False)
                        output = model(input_ids.to(model.llm.device)).logits[:, -1]
                        indices = torch.tensor([tokenizer.encode(letter)[1] for letter in letters]).to(model.llm.device)
                        output = torch.softmax(output, dim=-1)[:, indices]
                        ref_token = letters.index(question["Answer"])
                        ref_prob = output[:, ref_token].item()
                        entropy = - (output * torch.log(output)).sum().item()
                else:
                    prompt = question["Question"]
                    if "probe" in args.testfile:
                        prompt = question["Question"] + "Answer Yes or No directly."
                    conversation = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    input_ids = tokenizer.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        if args.nsamples > 0:
                            output = model(input_ids.to(model.llm.device)).logits[:, -1]
                            yesno_token = [tokenizer.encode("Yes")[1], tokenizer.encode("No")[1]]
                            if output.topk(1)[1][0, 0].item() in yesno_token:
                                sample_text = torch.softmax(output, dim=-1)[:, yesno_token][0].tolist()
                            else:
                                sample_texts = []
                                input_ids = input_ids.to(model.llm.device)
                                for t in range(args.nsamples):
                                    _, sample_text = model.generate(input_ids, do_sample=True, max_new_tokens=8)
                                    sample_texts.append(sample_text)
                                sample_text = sample_texts
                        else:
                            _, sample_text = model.generate(input_ids.to(model.llm.device), do_sample=False)
                        entropy = 0
                        ref_prob = 0
                result = {"question": question["Question"], "ref": question["Answer"], "pred": sample_text, "entropy": entropy, "acc_prob": ref_prob}
                if "Choices" in question:
                    if "False_in" in question:
                        result["False_in"] = question["False_in"]
                    result["Choices"] = question["Choices"]
                    result["Choice_distribution"] = {letters[i]: prob for i, prob in enumerate(output[0].tolist())}
                results[name].append(result)

        outfilename = args.outfile
        if args.origmodel:
            outfilename = outfilename.replace(".json", "_orig.json")
        if "mcq" in args.testfile:
            outfilename = outfilename.replace(".json", "_mcq.json")
        with open(outfilename, "w") as fout:
            json.dump(results, fout, indent=4)

    if args.do_selfcheck:
        selfcheckmodel = SelfCheckModel().to(device)
        selfcheckmodel.eval()
        selfcheckresults = {}
        for name in testdata.keys():
            if name not in selected_names:
                continue
            # attributes = namedict[name]
            logging(f"Generating selfcheck samples for {name}", args.logfile)
            # prompt = f"Your task is to generate accurate information about {name} covering these attributes: {attributes}. Create a single passage about {name} including all those attributes.\n\nYour passage:"
            prompt = f"Generate a passage about {name}."
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
    parser.add_argument(
        "--do_selfcheck",
        action='store_true',
        help="Run selfcheck score",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=0,
        help="Number of samples to draw",
    )
    args = parser.parse_args()
    main(args)
