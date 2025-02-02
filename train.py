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
from dataloader import SupervisedDataset, SupervisedWHPDataset, collate_fn, get_hallucinated_sample, SupervisedMCQDataset


accelerator = Accelerator()
device = accelerator.device
random.seed(1)
torch.manual_seed(1)

letters = ["A", "B", "C", "D", "E"]


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

def main(rank, args, world_size):
    # Save model configuration
    with open(os.path.join(args.outputdir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    os.system("cp {} {}".format(args.selected_ids, args.outputdir))

    # Define model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir="/home/gs534/rds/hpc-work/work/ckpts/",
    )
    with open(args.lora_config) as fin:
        lora_config = json.load(fin)
    os.system("cp {} {}".format(args.lora_config, os.path.join(args.outputdir, 'lora_config.json')))
    model = UnlearnModel(
        args.model_path,
        tokenizer,
        lora_rank=lora_config["lora_rank"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        lora_module=lora_config["lora_module"],
    )
    if args.load_from != "":
        modelpath = os.path.join(args.load_from, "pytorch_model.pt")
        trained_params = torch.load(modelpath)
        msg = model.load_state_dict(trained_params, strict=False)

    selfcheckmodel = None
    if "selfcheck" in args.losstype:
        selfcheckmodel = SelfCheckModel()
        selfcheckmodel.eval()

    ## Initialise data
    if "mcq" in args.losstype:
        traindata = SupervisedMCQDataset(
            args.train_data_path,
            args.prompt_path,
            tokenizer,
            selected_id=args.selected_ids if args.selected_ids != "" else args.selected_id,
            mem_mcq="mem" in args.losstype,
        )
    elif "whp" in args.losstype:
        traindata = SupervisedWHPDataset(
            args.train_data_path,
            args.prompt_path,
            tokenizer,
            n_passages=args.selfchecksamples,
            selected_id=args.selected_ids if args.selected_ids != "" else args.selected_id,
        )
        traindata.get_teacher_data(model.eval().to(device))
    else:
        traindata = SupervisedDataset(
            args.train_data_path,
            args.prompt_path,
            tokenizer,
            selected_id=args.selected_id,
            iterations=args.iterations,
        )
    train_dataloader = DataLoader(
        traindata,
        batch_size=args.batch_size,
        shuffle=True,
        # sampler=DistributedSampler(traindata),
        collate_fn=collate_fn,
    )

    ## Initialise criterion and optimiser
    if "selfcheck" in args.losstype:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    ## Optimiser
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if accelerator.state.deepspeed_plugin is None or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = accelerate.utils.DummyOptim(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = args.num_warmup_steps * max_train_steps

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    else:
        lr_scheduler = accelerate.utils.DummyScheduler(
            optimizer, total_num_steps=max_train_steps, warmup_num_steps=num_warmup_steps
        )

    if "selfcheck" in args.losstype:
        model, optimizer, train_dataloader, lr_scheduler, selfcheckmodel = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler, selfcheckmodel)
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler)

    print("Start training")
    best_val_loss = 10000
    for epoch in range(args.num_train_epochs):
        model.train()
        # train_dataloader.sampler.set_epoch(epoch)
        model = train_one_epoch(
            args,
            epoch,
            model,
            train_dataloader,
            traindata,
            optimizer,
            lr_scheduler,
            criterion,
            tokenizer,
            rank,
            world_size,
            selfcheckmodel,
            selected_name=traindata.selected_names,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        # torch.distributed.reduce(val_loss, 0)
        # torch.distributed.reduce(val_acc, 0)
        # Save models
        if accelerator.is_main_process:
            logging(f"Epoch {epoch} | Learning rate: {current_lr}", args.logfile)
            # save_checkpoint(model, tokenizer, args.outputdir, epoch)
        save_checkpoint(model, tokenizer, args.outputdir, epoch, "final")
        if "mcq" in args.losstype:
            eval_sample_mcq(args, model, traindata)


def save_checkpoint(model, tokenizer, outputdir, epoch, step):
    fulloutput = os.path.join(outputdir, "checkpoint.{}.{}".format(epoch, step))
    os.system(f"mkdir -p {fulloutput}")
    checkpoint = OrderedDict()
    for k, v in model.named_parameters():
        if v.requires_grad:
            checkpoint[k] = v
    torch.save(checkpoint, f'{fulloutput}/pytorch_model.pt')
    # save tokenizer
    tokenizer.save_pretrained(fulloutput)
    # save configuration
    model.llm.config.save_pretrained(fulloutput)
    return checkpoint


def gen_mem_sample(mem_samples, model):
    # Generate the sample to memorize
    mem_sample_ids = []
    mem_labels = []
    for mem_sample in mem_samples:
        mem_sample_id, mem_sample_text = model.generate(mem_sample, memorize=True)
        mem_sample_ids.append(torch.cat([mem_sample, mem_sample_id], dim=-1)[0])
        mem_labels.append(torch.cat([mem_sample*0-100, mem_sample_id], dim=-1)[0])
    mem_sample_id = pad_sequence(mem_sample_ids, batch_first=True, padding_value=0)
    mem_labels = pad_sequence(mem_labels, batch_first=True, padding_value=-100)
    return mem_sample_id, mem_labels

def train_one_epoch(
    args,
    epoch,
    model,
    train_dataloader,
    traindata,
    optimizer,
    lr_scheduler,
    criterion,
    tokenizer,
    rank,
    world_size,
    selfcheckmodel,
    selected_name="",
):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start = time.time()
    for i, batch in enumerate(train_dataloader):
        forget_samples, mem_samples = batch
        forget_sample_ids = []
        forget_labels = []
        forget_right_sample_ids = []
        forget_right_labels = []
        forget_logps = []
        # Resample passage
        if i % args.resample_frequency == 0 and "mcq" not in args.losstype and "whp" not in args.losstype:
            logging("="*89, args.logfile)
            logging("Resample at step: {}".format(i), args.logfile)
            with torch.no_grad():
                # First, test if the forget model is functioning well
                eval_sample(args, model, traindata, selected_name)
                # Generate the sample to forget
                for forget_sample in forget_samples:
                    if "selfcheck" in args.losstype:
                        logging("Generating selfcheck samples", args.logfile)
                        sample_passages = []
                        for k in range(args.selfchecksamples):
                            temp_schedule_reduce = min(0.5, (i // args.resample_frequency) * 0.1)
                            forget_sample_id, forget_sample_text = model.generate(forget_samples[0], temperature=1.5-temp_schedule_reduce)
                            sample_passages.append(forget_sample_text)
                            forget_sample_ids.append(torch.cat([forget_sample, forget_sample_id], dim=-1)[0])
                            forget_labels.append(torch.cat([forget_sample*0-100, forget_sample_id], dim=-1)[0])
                        selfcheckscores = selfcheckmodel.selfcheck(sample_passages)
                        logging("="*89, args.logfile)
                        logging("SelfCheckGPT score: {:.2f}".format(selfcheckscores.mean()*100), args.logfile)
                        logging("="*89, args.logfile)
                    else:
                        forget_sample_id, forget_sample_text = model.generate(forget_sample, memorize=True)
                        forget_right_sample_ids.append(torch.cat([forget_sample, forget_sample_id], dim=-1)[0])
                        forget_right_labels.append(torch.cat([forget_sample*0-100, forget_sample_id], dim=-1)[0])
                        if "rewrite" in args.losstype:
                            hallu_input = get_hallucinated_sample(forget_sample_text, selected_name, tokenizer).to(forget_sample.device)
                            forget_sample_id, forget_sample_text = model.generate(hallu_input, memorize=True)
                        forget_sample_ids.append(torch.cat([forget_sample, forget_sample_id], dim=-1)[0])
                        logging("="*89, args.logfile)
                        logging("Fake passage:", args.logfile)
                        logging(forget_sample_text, args.logfile)
                        forget_labels.append(torch.cat([forget_sample*0-100, forget_sample_id], dim=-1)[0])
                        if "kl" in args.losstype:
                            forget_sample_logp = model(forget_sample_ids[-1].unsqueeze(0), memorize=True).logits
                            forget_sample_logp = torch.softmax(forget_sample_logp, dim=-1)
                            forget_logps.append(forget_sample_logp[0])
                forget_sample_id = pad_sequence(forget_sample_ids, batch_first=True, padding_value=0)
                forget_labels = pad_sequence(forget_labels, batch_first=True, padding_value=-100)
                if "dpo" in args.losstype:
                    forget_right_sample_ids = pad_sequence(forget_right_sample_ids, batch_first=True, padding_value=0)
                    forget_right_labels = pad_sequence(forget_right_labels, batch_first=True, padding_value=-100)
                if "kl" in args.losstype:
                    forget_logps = pad_sequence(forget_logps, batch_first=True, padding_value=0)
                # Generate the sample to memorize
                mem_sample_id, mem_labels = gen_mem_sample(mem_samples, model)
            logging("="*89, args.logfile)
        elif "mcq" in args.losstype:
            forget_samples = [sample[0] for sample in forget_samples]
            forget_sample_id = pad_sequence(forget_samples, batch_first=True, padding_value=0)
            mem_sample_id, mem_labels = gen_mem_sample(mem_samples, model)
        elif "whp" in args.losstype and i == 0:
            with torch.no_grad():
                forget_sample_id, forget_sample_text = model.generate(forget_sample, memorize=True)
                forget_right_sample_ids.append(torch.cat([forget_sample, forget_sample_id], dim=-1)[0])
                forget_right_labels.append(torch.cat([forget_sample*0-100, forget_sample_id], dim=-1)[0])
                obfuscate_names = random.sample(traindata.mem_names, k=args.selfchecksamples)
                for selected_name in obfuscate_names:
                    hallu_input = get_hallucinated_sample(forget_sample_text, selected_name, tokenizer, whp=True).to(forget_sample.device)
                    forget_sample_id, forget_sample_text = model.generate(hallu_input, memorize=True)
                    forget_sample_ids.append(torch.cat([forget_sample, forget_sample_id], dim=-1)[0])
                    forget_labels.append(torch.cat([forget_sample*0-100, forget_sample_id], dim=-1)[0])
                    if "kl" in args.losstype:
                        forget_sample_logp = model(forget_sample_ids[-1].unsqueeze(0), memorize=True).logits
                        forget_sample_logp = torch.softmax(forget_sample_logp, dim=-1)
                        forget_logps.append(forget_sample_logp[0])
                logging("="*89, args.logfile)
                logging("Fake passage:", args.logfile)
                logging(forget_sample_text, args.logfile)
                mem_sample_id = None

        # Forward
        indices = torch.tensor([tokenizer.encode(letter)[1] for letter in letters]).to(model.llm.device)
        if mem_sample_id is not None:
            mem_output = model(mem_sample_id).logits[:, :-1]
            if "bothflatten" in args.losstype:
                if "flattenO" in args.losstype:
                    with torch.no_grad():
                        mem_orig_output = model(mem_sample_id, memorize=True).logits[:, -2]
                        mem_orig_output = torch.softmax(mem_orig_output[:, indices], dim=-1)
                        mem_orig_output = mem_orig_output.data
                    mem_output = torch.log_softmax(mem_output[:, -1, indices], dim=-1)
                    loss_mem = - mem_orig_output * mem_output
                    loss_mem = loss_mem.sum(dim=-1).mean() * args.retain_factor
                else:
                    with torch.no_grad():
                        mem_orig_output = model(mem_sample_id, memorize=True).logits[:, -2]
                        mem_orig_output = torch.softmax(mem_orig_output, dim=-1)
                        mem_orig_output = mem_orig_output.data
                    mem_output = torch.log_softmax(mem_output[:, -1], dim=-1)
                    loss_mem = - mem_orig_output * mem_output
                    loss_mem = loss_mem.sum(dim=-1).mean() * args.retain_factor
            else:
                loss_mem = criterion(mem_output.reshape(-1, mem_output.size(-1)), mem_labels[:, 1:].reshape(-1)) * args.retain_factor
            loss_mem = loss_mem.mean()
        else:
            loss_mem = 0

        min_step = 10
        if "selfcheck" in args.losstype and args.selfchecksamples > min_step:
            accelerator.backward(loss_mem.mean())
            forget_output = []
            loss = 0
            for step in range(0, args.selfchecksamples, min_step):
                forget_output_single = model(forget_sample_id[step:step+min_step]).logits[:, :-1]
                forget_labels_single = forget_labels[step:step+min_step, 1:]
                forget_output.append(forget_output_single)
                loss_forget = criterion(
                    forget_output_single.reshape(-1, forget_output_single.size(-1)), forget_labels_single.reshape(-1))
                loss_forget = loss_forget.sum(dim=-1) / (forget_labels_single != -100).sum(dim=-1) * selfcheckscores[step:step+min_step]
                loss_forget = loss_forget.sum() / args.selfchecksamples
                accelerator.backward(loss_forget)
                loss += loss_forget.item()
        else:
            forget_output = model(forget_sample_id).logits
            forget_output = forget_output[:, -1] if "mcq" in args.losstype else forget_output[:, :-1]
            if "kl" in args.losstype:
                loss_mask = forget_labels[:, 1:] != -100
                forget_output_logp = torch.log_softmax(forget_output, dim=-1)
                loss_forget = - (forget_logps[:, 1:] * forget_output_logp).sum(dim=-1) * loss_mask
                loss_forget = loss_forget.sum() / loss_mask.sum()
            if "mcq" in args.losstype:
                if "flattenO" in args.losstype:
                    forget_output = - torch.log_softmax(forget_output[:, indices], dim=-1)
                    loss_forget = (forget_output).mean()
                elif "accum" in args.losstype:
                    losses_forget = []
                    for choice in ["A", "B", "C", "D", "E"]:
                        label = [tokenizer.encode(choice)[1]]
                        loss_forget = - torch.log_softmax(forget_output, dim=-1)[:, label]
                        losses_forget.append(loss_forget)
                    loss_forget = torch.concat(losses_forget, dim=0).mean()
                elif "flatten" in args.losstype:
                    forget_output = - torch.log_softmax(forget_output, dim=-1)[:, indices]
                    loss_forget = (forget_output).mean()
                else:
                    random_choices = random.choices(["A", "B", "C", "D", "E"], k=forget_output.size(0))
                    label = [tokenizer.encode(random_choice)[1] for random_choice in random_choices]
                    loss_forget = - torch.log_softmax(forget_output, dim=-1)[:, label]
                    loss_forget = loss_forget.mean()
            else:
                loss_forget = criterion(forget_output.reshape(-1, forget_output.size(-1)), forget_labels[:, 1:].reshape(-1))
            if args.losstype == "ga":
                # Negative loss - Gradient Ascent
                loss = - loss_forget
            elif args.losstype in ["rewrite", "rewritekl"] or "mcq" in args.losstype:
                loss = loss_forget
            elif args.losstype == "rewritedpo":
                forget_right_output = model(forget_right_sample_ids).logits[:, :-1]
                loss_forget_right = criterion(forget_right_output.view(-1, forget_right_output.size(-1)), forget_right_labels[:, 1:].reshape(-1))
                win_seqlen = (forget_labels[:, 1:] != -100).sum()
                lose_seqlen = (forget_right_labels[:, 1:] != -100).sum()
                with torch.no_grad():
                    forget_output_ref = model(forget_sample_id, memorize=True).logits[:, :-1]
                    forget_logp_ref = criterion(forget_output_ref.view(-1, forget_output_ref.size(-1)), forget_labels[:, 1:].reshape(-1))
                    forget_right_output_ref = model(forget_right_sample_ids, memorize=True).logits[:, :-1]
                    forget_right_logp_ref = criterion(forget_right_output_ref.view(-1, forget_right_output_ref.size(-1)), forget_right_labels[:, 1:].reshape(-1))
                win_logp = win_seqlen * (forget_logp_ref - loss_forget)
                lose_logp = lose_seqlen * (forget_right_logp_ref - loss_forget_right)
                loss = - torch.nn.functional.logsigmoid(args.npo_beta * (win_logp - lose_logp))
            elif args.losstype == "npo":
                # Negative Preference Optimization
                seqlen = (forget_labels[:, 1:] != -100).sum()
                with torch.no_grad():
                    forget_output_ref = model(forget_sample_id, memorize=True).logits[:, :-1]
                    forget_logp_ref = criterion(forget_output_ref.view(-1, forget_output_ref.size(-1)), forget_labels[:, 1:].reshape(-1))
                loss = - 2 / args.npo_beta * torch.nn.functional.logsigmoid(- args.npo_beta * seqlen * (forget_logp_ref-loss_forget))
            loss = loss + loss_mem

            loss = loss / args.gradient_accumulation_steps
            # loss.backward()
            accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if (i + 1) % args.log_interval == 0 and accelerator.is_main_process:
            elasped_time = time.time() - start
            if args.losstype == "selfcheck":
                logging(f"Epoch {epoch} | Batch {i}/{trainsize} | Loss: {loss} | time {elasped_time}", args.logfile)
            elif "mcq" in args.losstype:
                logging(f"Epoch {epoch} | Batch {i}/{trainsize} | Loss: {loss_forget} | Loss mem: {loss_mem/args.retain_factor} | time {elasped_time}", args.logfile)
            else:
                PPL = math.exp(loss_forget.item() * args.gradient_accumulation_steps)
                PPL_mem = math.exp(loss_mem.item() * args.gradient_accumulation_steps)
                logging(f"Epoch {epoch} | Batch {i}/{trainsize} | PPL forget: {PPL} | PPL mem: {PPL_mem} | time {elasped_time}", args.logfile)
        if (i + 1) % args.save_interval == 0 and accelerator.is_main_process:
            logging(f"Saving at Step {i+1}", args.logfile)
            save_checkpoint(model, tokenizer, args.outputdir, epoch, i+1)
    return model


def eval_sample(
    args,
    model,
    traindata,
    selected_name="",
):
    i = random.randint(1, len(traindata.prompt_bank["eval_prompts"])-1)
    logging("="*89, args.logfile)
    logging("Sampeld passage {}".format(i), args.logfile)
    logging("="*89, args.logfile)
    input_ids = traindata.get_new_prompt(i)
    _, forget_sample_text = model.generate(input_ids.to(model.llm.device))
    logging(forget_sample_text, args.logfile)


def eval_sample_mcq(
    args,
    model,
    traindata,
):
    input_ids, name = traindata.get_new_prompt()
    _, forget_sample_text = model.generate(input_ids.to(model.llm.device))
    logging("="*89, args.logfile)
    logging("Sampeld passage for{}".format(name), args.logfile)
    logging("-"*89, args.logfile)
    logging(forget_sample_text, args.logfile)
    logging("="*89, args.logfile)


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
        "--train_data_path",
        type=str,
        default="./hf_models",
        help="Path to the train data file",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./hf_models",
        help="Path to the prompt file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default='./exp/clip_vlm',
        help="Path to the output dir",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log interval",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="How many rounds of training to forget this person",
    )
    parser.add_argument(
        "--resample_frequency",
        type=int,
        default=1,
        help="How many rounds of training to forget this person",
    )
    parser.add_argument(
        "--selected_id",
        type=int,
        default=0,
        help="select which person to forget",
    )
    parser.add_argument(
        "--selected_ids",
        type=str,
        default="config/unlearn_ids.json",
        help="select which person to forget",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Saving interval",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default='12355',
        help="Master port number",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="data/lora_config.json",
        help="LoRA configuration",
    )
    parser.add_argument(
        "--losstype",
        type=str,
        default="ga",
        help="type of loss to train forget model",
    )
    parser.add_argument(
        "--load_from",
        type=str,
        default="",
        help="path to load checkpoint from",
    )
    parser.add_argument(
        "--npo_beta",
        type=float,
        default=1,
        help="NPO beta",
    )
    parser.add_argument(
        "--selfchecksamples",
        type=int,
        default=3,
        help="number of samples for SelfCheckGPT",
    )
    parser.add_argument(
        "--retain_factor",
        type=float,
        default=0.0,
        help="factor for the retain loss",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)
    # mp.spawn(main, args=(args, world_size,), nprocs=world_size)
    main(0, args, world_size)
