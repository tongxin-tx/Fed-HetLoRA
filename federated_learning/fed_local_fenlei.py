import numpy as np
import torch
import copy
from trl import SFTTrainer
from transformers import TrainerCallback, Trainer
from peft import get_peft_model_state_dict, set_peft_model_state_dict
# from datasets import load_dataset, load_metric
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score
# metric = load_metric("glue", "mrpc")
# def compute_metrics(eval_pred):
#     """
#     计算评估指标
#     """
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# accuracy_metric = evaluate.load("accuracy")
# f1_metric = evaluate.load("f1")

# 定义评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # predictions = torch.argmax(torch.tensor(logits), dim=-1).to('cuda')
    # labels=torch.tensor(labels).to('cuda')
    predictions = torch.argmax(torch.tensor(logits).cpu(), dim=-1).numpy()
    labels = torch.tensor(labels).cpu().numpy()
    accuracy = (predictions == labels).mean()
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# def compute_metrics(eval_pred):
#     """
#     计算评估指标
#     """
#     logits, labels = eval_pred
#     # 将 logits 和 labels 转换为 torch.Tensor
#     logits = torch.tensor(logits)
#     labels = torch.tensor(labels)
#     # 使用 torch.argmax 计算预测结果
#     predictions = torch.argmax(logits, dim=-1)
#     # 将预测结果和标签转换为 numpy 数组，以便使用 metric.compute
#     predictions = predictions.cpu().numpy()
#     labels = labels.cpu().numpy()
#     return metric.compute(predictions=predictions, references=labels)data_collator,
def get_fed_local_fenlei_trainer( model, tokenizer, training_args, local_dataset,test_dataset, global_dict, fed_args, script_args,local_auxiliary, global_auxiliary):
    # print(local_dataset[0]['input_ids'])
    # print(test_dataset[0])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=local_dataset,
        eval_dataset=test_dataset,
        # data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer