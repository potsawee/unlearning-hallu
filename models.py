import argparse
import logging
import math
import os
from time import time
from copy import deepcopy
import random
import json
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import spacy
import six
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel


nlp = spacy.load("en_core_web_sm")

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

    def generate(self, inputs, memorize=False, temperature=1.0, do_sample=True):
        attention_mask = torch.ones_like(inputs)
        if memorize:
            self.llm.disable_adapters()
        generate_ids = self.llm.generate(
            inputs,
            max_new_tokens=512,
            attention_mask=attention_mask,
            temperature=temperature,
            top_p=0.9,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if memorize:
            self.llm.enable_adapters()
        generate_text = self.tokenizer.batch_decode(
            generate_ids[:, inputs.size(1):], skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return generate_ids[:, inputs.size(1):], generate_text


class SelfCheckModel(torch.nn.Module):
    def __init__(
        self,
        max_evidence=20,
        model_path="meta-llama/Llama-3.1-8B-Instruct",
    ):
        super(SelfCheckModel, self).__init__()
        self.max_evidence = max_evidence
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/data/milsrg1/huggingface/cache/gs534/cache",
            # attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir="/data/milsrg1/huggingface/cache/gs534/cache",
        )

    def selfcheck(self, passages, memorize=False):
        passage_scores = []
        for i, passage in tqdm(enumerate(passages)):
            print("Forwarding passage {}".format(i))
            sentences = [sent.text.strip() for sent in nlp(passage).sents]
            other_passages = passages[:i] + passages[i+1:]
            if len(other_passages) > self.max_evidence:
                other_passages = random.sample(other_passages, self.max_evidence)
            score = self.selfcheck_per_passage(passage, other_passages, memorize=memorize)
            passage_scores.append(score)
        return torch.stack(passage_scores)

    def selfcheck_per_passage(self, passage, sampled_passages, memorize=False):
        passage_split = [p for p in passage.split("\n") if p != ""]
        sentences = []
        for p in passage_split:
            sentences += [sent.text.strip() for sent in nlp(p).sents]
        prompt_template = """Context: {context}\n\nSentence: {sentence}\n\Does the sentence contradict the context above? Answer Yes or No.\n\nAnswer: """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        for sent_i in tqdm(range(num_sentences)):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 

                prompt = prompt_template.format(context=sample, sentence=sentence)
                messages = [
                    {"role": "user", "content": prompt}
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.llm.device)
                generate_ids = self.llm.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids[:, input_ids.size(1):], skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = output_text.replace(prompt, "").lower().lstrip()
                if generate_text[:3] == 'yes':
                    score_ = 1.0
                elif generate_text[:2] == 'no':
                    score_ = 0.0
                else:
                    score_ = 0.5
                scores[sent_i, sample_i] = score_
        scores = torch.tensor(scores).to(self.llm.device)
        # score = (scores.mean(dim=-1).float() > 0.5).float().mean()
        score = scores.mean()
        return score
