import copy
import os
import random
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
import accelerate
import math

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from numpy.linalg import svd
import json
import torch.nn as nn
from utils.process_dataset import process_mac_dataset
from utils.client_dection import compute,weight,julei
torch.manual_seed(42)



def matrix_factorization(A, rank, lr=0.001, e=0.1, epochs=500, patience=5):
    m, n = A.shape
    # 放大矩阵 A 以避免数值过小
    scale_factor = 100000
    A = scale_factor * A

    # 初始化 U 和 V 矩阵
    U = nn.Parameter(torch.zeros(m, rank, device=A.device))
    V = nn.Parameter(torch.zeros(rank, n, device=A.device))
    # 使用正态分布初始化 U 和 V
    # nn.init.normal_(U, mean=0, std=e)
    nn.init.zeros_(U)
    nn.init.normal_(V, mean=0, std=e)

    # 定义优化器
    optimizer = torch.optim.Adam([U, V], lr=lr)
    loss_fn = nn.MSELoss()

    # 记录最小损失和对应的参数
    best_loss = float('inf')
    best_U = U.data.clone()
    best_V = V.data.clone()
    # 记录损失连续上升的次数
    rising_count = 0

    for epoch in range(epochs):
        # 前向传播
        A_pred = torch.matmul(U, V)  # 矩阵乘法
        loss = loss_fn(A_pred, A)  # 计算误差

        # 反向传播
        optimizer.zero_grad()
        loss.backward()  # 计算梯度

        # 梯度裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_([U, V], max_norm=1.0)

        optimizer.step()  # 更新参数

        # 打印损失（每 100 次迭代）
        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

        # 检查损失是否上升
        if loss.item() > best_loss:
            rising_count += 1
            if rising_count >= patience:
                # print(f"Early stopping at epoch {epoch + 1} due to consecutive rising loss.")
                U.data = best_U
                V.data = best_V
                break
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
            # print(f"Loss increased at epoch {epoch + 1}, reducing learning rate to {param_group['lr']}")
        else:
            rising_count = 0
            best_loss = loss.item()
            best_U = U.data.clone()
            best_V = V.data.clone()

    U = U / 5000
    V = V / 20

    return U, V



def creat_W():
    weight_dict = {}

    # 层数和权重类型
    num_layers = 16  # 层数
    weight_types = ["q_proj", "v_proj"]

    # 构建字典
    for layer in range(num_layers):
        for weight_type in weight_types:
            key = f"base_model.model.model.layers.{layer}.self_attn.{weight_type}"
            weight_dict[key] = 0  # 或者 torch.randn(shape) 初始化为权重矩阵

    return weight_dict

# def w_huanyuan(weight_dict):


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)
print('123456',peft_config)
# ===== Load the dataset =====
rank=[4,8,16,8]

# script_args.dataset_name="medalpaca/medical_meadow_medical_flashcards"
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)
# script_args.dataset_name="PKU-Alignment/BeaverTails"#hf-datasets/WildChat"
# dataset2 = get_dataset("TIGER-Lab/MathInstruct", script_args.local_data_dir)
# dataset2 = process_sft_dataset("TIGER-Lab/MathInstruct", dataset2, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
# local_datasets2 = split_dataset(fed_args, script_args, dataset2)
# local_datasets3 = split_dataset(fed_args, script_args, dataset3)
print(len(local_datasets))
print('3333333333333333333333333333',len(local_datasets[0]))
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
# print(len(sample_num_list))


# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)
# device_map='auto'
device_map='cuda:0'
print('111111111111',device_map)
print(script_args.model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
models=[]
global_dict_list=[]


# for i in range(fed_args.num_clients):
peft_config.r=rank[0]
model0=get_peft_model(model,peft_config)


del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
peft_config.r=rank[1]
model1=get_peft_model(model,peft_config)

del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
peft_config.r=rank[2]

model2=get_peft_model(model,peft_config)

del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
peft_config.r=rank[3]
model3=get_peft_model(model,peft_config)
del model
torch.cuda.empty_cache()

models.append(model0)
models.append(model1)
models.append(model2)
models.append(model3)
# model = get_peft_model(model, peft_config)
models[0].print_trainable_parameters()
models[1].print_trainable_parameters()
models[2].print_trainable_parameters()
models[3].print_trainable_parameters()
models[0].config.use_cache = False  # silence the warnings. Please re-enable for inference!
models[1].config.use_cache = False
models[2].config.use_cache = False
models[3].config.use_cache = False
if training_args.gradient_checkpointing:
    models[0].enable_input_require_grads()
    models[1].enable_input_require_grads()
    models[2].enable_input_require_grads()
    models[3].enable_input_require_grads()

# del models[2]
# del models[3]
# torch.cuda.empty_cache()
global_dict_list.append(copy.deepcopy(get_peft_model_state_dict(models[0])))
global_dict_list.append(copy.deepcopy(get_peft_model_state_dict(models[1])))
global_dict_list.append(copy.deepcopy(get_peft_model_state_dict(models[2])))
global_dict_list.append(copy.deepcopy(get_peft_model_state_dict(models[3])))
# print(global_dict_list[i].keys())
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict_list[0])

global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict_list[0])

# ===== Define the global and local models =====
# global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict_list[i]) for i in range(fed_args.num_clients)]

linshi_w=creat_W()
local_dict_list_w = [copy.deepcopy(linshi_w) for i in range(fed_args.num_clients)]
global_dict_w =copy.deepcopy(creat_W())

del model0
del model1
del model2
del model3
del models
# del models[0]
# del models[0]
# del models[0]
# del models[0]
torch.cuda.empty_cache()

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna
tokenizer.pad_token = tokenizer.eos_token
# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
test_loss = [[] for i in range(fed_args.num_clients)]


for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")

    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        peft_config.r = rank[client]
        # model=copy.deepcopy(model1)
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if script_args.load_in_8bit or script_args.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, peft_config)
        set_peft_model_state_dict(model, global_dict_list[client])   # sync the global model to the local model
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args,
                                             script_args)  # get the required sub-dataset for this round

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset.train_test_split(test_size=0.2)['train'],
            test_dataset=sub_dataset.train_test_split(test_size=0.2)['test'],
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict_list[client],
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary
        )


        results = trainer.train()
        test_results = trainer.evaluate()
        print(test_results)
        training_loss[client].append(results.training_loss)
        test_loss[client].append(test_results['eval_loss'])
        # test_loss[client].append(test_results['eval_loss'])

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))  # copy is needed!
        del model
        torch.cuda.empty_cache()

        for key in global_dict_list[client].keys():
            if "lora_A.weight" in key:
                # 找到对应的 lora_B
                layer_name = key.replace(".lora_A.weight", "")  # 提取层的名称
                lora_A = local_dict_list[client][key]
                lora_B_key = key.replace(".lora_A.weight", ".lora_B.weight")
                # print(lora_B_key)
                if lora_B_key in local_dict_list[client]:
                    lora_B = local_dict_list[client][lora_B_key]
                    local_dict_list_w[client][layer_name] = torch.matmul(lora_B,lora_A)


    # ===== Server aggregates the local models =====

    global_dict_w, global_auxiliary = global_aggregate(
        fed_args, global_dict_w, local_dict_list_w, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    for client in range(fed_args.num_clients):
        # print(rank[client])
        for i in range(len(global_dict_w)):
            layer_base = list(global_dict_w.keys())[i]
            B, A = matrix_factorization(global_dict_w[layer_base], rank[client],0.01,1, 10000,5)
            global_dict_list[client][f"{layer_base}.lora_A.weight"] = A
            global_dict_list[client][f"{layer_base}.lora_B.weight"] = B
    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))

    print(training_loss)
    print(test_loss)
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    np.save(os.path.join(script_args.output_dir, "test_loss.npy"), np.array(test_loss))

    torch.cuda.empty_cache()