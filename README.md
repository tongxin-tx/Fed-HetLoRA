#  Fed-HetLoRA: A Federated Heterogeneous LoRA Fine-Tuning method for Large Language Models

**Fed-HetLoRA** is an open source research codebase for training Large Language Models (LLMs) through federated learning, supporting heterogeneous LoRA training.

![framework](FL-LoRA.pdf)

## Requirements
pip install -r requirements.txt

The training script is in `training_scripts/run_sft.sh`.
## Federated Instruction Tuning

### Heterogeneous

```
CUDA_VISIBLE_DEVICES=1 heterogeneous_Instruction_tuning.py \
 --model_name_or_path "meta-llama/Llama-3.2-3B" \
 --dataset_name "vicgalle/alpaca-gpt4" \
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients 4 \
 --sample_clients 4 \
 --max_steps 10 \
 --num_rounds 200 \
 --batch_size 16 \
 --gradient_accumulation_steps 1 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --load_in_8bit \
 --output_dir "./output" \
 --template "alpaca" \
```

### Homogenous

```
CUDA_VISIBLE_DEVICES=1 homogenous_Instruction_tuning.py \
 --model_name_or_path "meta-llama/Llama-3.2-3B" \
 --dataset_name "vicgalle/alpaca-gpt4" \
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients 4 \
 --sample_clients 4 \
 --max_steps 10 \
 --num_rounds 200 \
 --batch_size 16 \
 --gradient_accumulation_steps 1 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --load_in_8bit \
 --output_dir "./output" \
 --template "alpaca" \
```

## Federated Classification

### Heterogeneous

```
CUDA_VISIBLE_DEVICES=1 heterogeneous_classification.py \
 --model_name_or_path "meta-llama/Llama-3.2-3B" \
 --dataset_name "/home/dell/tx_project/gitee-ai/hub/datasets--imdb" \
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients 4 \
 --sample_clients 4 \
 --max_steps 10 \
 --num_rounds 200 \
 --batch_size 32 \
 --gradient_accumulation_steps 1 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --load_in_8bit \
 --output_dir "./output" \
 --template "alpaca" \
```

### Homogenous

```
CUDA_VISIBLE_DEVICES=1 homogenous_classification.py \
 --model_name_or_path "meta-llama/Llama-3.2-3B" \
 --dataset_name "/home/dell/tx_project/gitee-ai/hub/datasets--imdb" \
 --dataset_sample 20000 \
 --fed_alg "fedavg" \
 --num_clients 4 \
 --sample_clients 4 \
 --max_steps 10 \
 --num_rounds 200 \
 --batch_size 32 \
 --gradient_accumulation_steps 1 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --load_in_8bit \
 --output_dir "./output" \
 --template "alpaca" \
```


