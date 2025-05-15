#!/bin/bash

max_steps=10
num_rounds=20
batch_size=32
gradient_accumulation_steps=1
seq_length=512
num_clients=4

sample_clients=4
lora_r=4
lora_alpha=64   # twice of lora_r #16
lr=5e-5

#local_data_dir="data1/jinni/first/OpenFedLLM-main/vicgalle/alpaca-gpt4/"#hf-datasets/WildChat  openai-community/gpt2  Llama-2-7b-hf    # you may uncomment this line if your data is stored locally and include it in the python command
#dataset_name="vicgalle/alpaca-gpt4""vicgalle/alpaca-gpt4" "medalpaca/medical_meadow_medical_flashcards"
#dataset_names=("vicgalle/alpaca-gpt4" "FinGPT/fingpt-sentiment-train" "medalpaca/medical_meadow_medical_flashcards")
dataset_names=("/home/dell/tx_project/gitee-ai/hub/datasets--imdb")
dataset_sample=20000
model_name_or_path="meta-llama/Llama-3.2-3B"
output_dir=./output/yigou-3B



gpu=0
fed_alg="fedavg"
for dataset_name in "${dataset_names[@]}"; do

  CUDA_VISIBLE_DEVICES=$gpu python my-dofit-rank-fenlei-client=4-zero.py \
   --learning_rate $lr \
   --model_name_or_path $model_name_or_path \
   --dataset_name $dataset_name \
   --dataset_sample $dataset_sample \
   --fed_alg $fed_alg \
   --num_clients $num_clients \
   --sample_clients $sample_clients \
   --max_steps $max_steps \
   --num_rounds $num_rounds \
   --batch_size $batch_size \
   --gradient_accumulation_steps $gradient_accumulation_steps \
   --seq_length $seq_length \
   --peft_lora_r $lora_r \
   --peft_lora_alpha $lora_alpha \
   --use_peft \
   --load_in_8bit \
   --output_dir $output_dir \
   --template 'alpaca'
done