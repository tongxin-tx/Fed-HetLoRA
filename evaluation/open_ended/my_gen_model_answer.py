import random

import datasets
import argparse
import json
import sys

from datasets import load_dataset

sys.path.append("../../")
from tqdm import tqdm
import os
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
# from vllm import LLM, SamplingParams

from utils.template import TEMPLATE_DICT
parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--template", type=str, default="alpaca")
parser.add_argument("--use_vllm", action="store_true")
parser.add_argument("--bench_name", type=str, default="vicuna")
args = parser.parse_args()
print(args)
template = TEMPLATE_DICT[args.template][0]
print(f">> You are using template: {template}")

eval_set = datasets.load_dataset("json", data_files="output.json")['train']
def rename(example):
    example['instruction'] = example['turns']#[0]
    return example
eval_set = eval_set.map(rename)
max_new_tokens = 2048



if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str                       # mainly for base model


# ============= Load previous results if exists =============
result_path = f"{args.bench_name}_model_answer_{model_name}.json"

if os.path.exists(result_path):
    with open(result_path, "r") as f:
        result_list = json.load(f)
else:
    result_list = []
existing_len = len(result_list)
print(f">> Existing length: {existing_len}")



device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to(device)
if args.lora_path is not None:
    model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16).to(device)
print('66666666666666666666',model_name)
#model_name='vicgalle/alpaca-gpt4'
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False)#args.model_nam

for i, example in tqdm(enumerate(eval_set)):
    if i < existing_len:
        continue
    if args.bench_name == "advbench":
        instruction = template.format(example["instruction"]+'.', "", "")[:-1]
    else:
        instruction = template.format(example["instruction"], "", "")[:-1]      # TODO: use fastchat conversation
    input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
    output_ids = model.generate(inputs=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=1.0, temperature=0.7)
    output_ids = output_ids[0][len(input_ids[0]):]
    result = tokenizer.decode(output_ids, skip_special_tokens=True)
    example['output'] = result
    example['generator'] = model_name

    print(f"\nInput: \n{instruction}")
    print(f"\nOutput: \n{result}")
    print("="*100)
    result_list.append(example)
    with open(result_path, "w") as f:
        json.dump(result_list, f, indent=4)

print(f">> You are using template: {template}")


# macdataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
# print(macdataset[1])
# macdataset = macdataset.rename_column("prompt", "turns")
# macdataset = macdataset.filter(lambda x: x['is_safe'] == False)
# macdataset = macdataset.remove_columns(['category', 'is_safe','response'])
#
#
#
# data_list = macdataset.to_list()
# print(len(data_list))
# # 打开JSON文件，准备写入
# with open('output.json', 'w') as f:
#     json.dump(data_list, f, indent=4)
#
# # eval_set = datasets.load_dataset("json", data_files="data/vicuna/question1.jsonl")['train']
#

