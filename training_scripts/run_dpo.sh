max_steps=10
num_rounds=100
batch_size=8
gradient_accumulation_steps=1
seq_length=512
num_clients=5
sample_clients=2
lr=5e-5

#local_data_dir="/data1/jinni/first/OpenFedLLM-main/output/alpaca-gpt4_20000_fedavg_c5s2_i10_b16a1_l512_r16a64_20241112224801/checkpoint-100"       # you may uncomment this line if your data is stored locally and include it in the python command
#dataset_name="Anthropic/hh-rlhf"
dataset_name="HuggingFaceH4/ultrafeedback_binarized"
dataset_sample=20000
#model_name_or_path="meta-llama/Llama-2-7b-hf"
model_name_or_path="/data1/jinni/first/OpenFedLLM-main/output/alpaca-gpt4_sft_clean_100/full-100"
#"cognitivecomputations/Wizard-Vicuna-7B-Uncensored"
#"ehartford/Wizard-Vicuna-7B-Uncensored"
output_dir=./output

gpu=2
fed_alg="fedavg"

CUDA_VISIBLE_DEVICES=$gpu python main_dpo.py \
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
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "vicuna_v1.1"