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

    ## Initialise data
    traindata = SupervisedWHPDataset(
        args.train_data_path,
        args.prompt_path,
        tokenizer,
        n_passages=args.selfchecksamples,
        selected_id=args.selected_ids if args.selected_ids != "" else args.selected_id,
        obfuscate_passages=args.obfuscate_passages,
        passage_id=args.passage_id,
    )
    traindata.get_teacher_data(model.eval().to(device), args.outputdir)
    train_dataloader = DataLoader(
        traindata,
        batch_size=args.batch_size,
        shuffle=True,
        # sampler=DistributedSampler(traindata),
        collate_fn=collate_fn,
    )

    ## Initialise criterion and optimiser
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
        )
        current_lr = optimizer.param_groups[0]["lr"]
        # torch.distributed.reduce(val_loss, 0)
        # torch.distributed.reduce(val_acc, 0)
        # Save models
        if accelerator.is_main_process:
            logging(f"Epoch {epoch} | Learning rate: {current_lr}", args.logfile)
            # save_checkpoint(model, tokenizer, args.outputdir, epoch)
        save_checkpoint(model, tokenizer, args.outputdir, epoch, "final")
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
):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start = time.time()
    for i, batch in enumerate(train_dataloader):
        forget_samples, forget_labels, forget_dist = batch

        # Forward
        if args.retain_factor > 0 and mem_sample_id is not None:
            mem_output = model(mem_sample_id).logits
            mem_output = mem_output[:, :-1]
            loss_mem = criterion(mem_output.reshape(-1, mem_output.size(-1)), mem_labels[:, 1:].reshape(-1)) * args.retain_factor
            loss_mem = loss_mem.mean()
        else:
            loss_mem = 0

        forget_output = model(forget_samples).logits[:, :-1]
        if "kl" in args.losstype:
            forget_output = torch.log_softmax(forget_output, dim=-1)
            loss_forget = forget_dist[:, :-1] * forget_output
            loss_mask = forget_labels[:, 1:] != -100
            loss_forget = (loss_forget.sum(dim=-1) * loss_mask).sum() / loss_mask.sum()
        else:
            loss_forget = criterion(forget_output.reshape(-1, forget_output.size(-1)), forget_labels[:, 1:].reshape(-1))
        loss = loss_forget + loss_mem
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if (i + 1) % args.log_interval == 0 and accelerator.is_main_process:
            elasped_time = time.time() - start
            PPL = math.exp(loss_forget.item() * args.gradient_accumulation_steps)
            PPL_mem = math.exp(loss_mem.item() * args.gradient_accumulation_steps)
            logging(f"Epoch {epoch} | Batch {i}/{trainsize} | PPL forget: {PPL} | PPL mem: {PPL_mem} | time {elasped_time}", args.logfile)
        if (i + 1) % args.save_interval == 0 and accelerator.is_main_process:
            logging(f"Saving at Step {i+1}", args.logfile)
            save_checkpoint(model, tokenizer, args.outputdir, epoch, i+1)
    return model


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
    parser.add_argument(
        "--obfuscate_passages",
        type=str,
        default="",
        help="samples to obfuscate",
    )
    parser.add_argument(
        "--passage_id",
        type=str,
        default="-1",
        help="only used when 1 passage is sampled",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)
    # mp.spawn(main, args=(args, world_size,), nprocs=world_size)
    main(0, args, world_size)

