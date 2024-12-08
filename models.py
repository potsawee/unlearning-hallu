import argparse
import logging
import math
import os
from time import time
from copy import deepcopy
import random
import json
from copy import deepcopy

import numpy as np
import six
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel


class UnlearnModel(torch.nn.Module):
    def __init__(
        self,
        model_path,
        tokenizer,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_module=["q_proj", "v_proj"],
        uselora=True,
    ):
        super(UnlearnModel, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/data/milsrg1/huggingface/cache/gs534/cache",
        )
        self.uselora = uselora
        if self.uselora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_module,
            )
            # self.llm = get_peft_model(self.llm, peft_config)
            self.llm.add_adapter(peft_config)
            self.llm.enable_adapters()
        self.tokenizer = tokenizer

    def forward(self, inputs, memorize=False):
        attention_mask = torch.ones_like(inputs)
        if memorize:
            self.llm.disable_adapters()
        outputs = self.llm(
            input_ids=inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        if memorize:
            self.llm.enable_adapters()
        return outputs

    def generate(self, inputs, memorize=False):
        attention_mask = torch.ones_like(inputs)
        if memorize:
            self.llm.disable_adapters()
        generate_ids = self.llm.generate(
            inputs,
            max_new_tokens=512,
            attention_mask=attention_mask,
            temperature=1.0,
            top_p=0.9,
            do_sample=True,
        )
        if memorize:
            self.llm.enable_adapters()
        generate_text = self.tokenizer.batch_decode(
            generate_ids[:, inputs.size(1):], skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return generate_ids[:, inputs.size(1):], generate_text
