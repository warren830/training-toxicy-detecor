import os
import numpy as np
import random
import json
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from typing import List, Tuple
import torch

# 2. hyperparameters定义
hyperparameters = {
    'learning_rate': float(os.environ.get('SM_HP_learning_rate', '2e-5')),
    'train_batch_size': int(os.environ.get('SM_HP_train_batch_size', '16')),
    'eval_batch_size': int(os.environ.get('SM_HP_eval_batch_size', '16')),
    'epochs': float(os.environ.get('SM_HP_epochs', '6')),
    'weight_decay': float(os.environ.get('SM_HP_weight_decay', '0.01')),
    'warmup_steps': int(os.environ.get('SM_HP_warmup_steps', '100')),
    'model_name': os.environ.get('SM_HP_model_name', 'thu-coai/roberta-base-cold'),
    'max_length': int(os.environ.get('SM_HP_max_length', '128')),
    'early_stopping_patience': int(os.environ.get('SM_HP_early_stopping_patience', '3')), # 增加耐心
    'dropout_rate': float(os.environ.get('SM_HP_dropout_rate', '0.3')),
    'label_smoothing': float(os.environ.get('SM_HP_label_smoothing', '0.1')),
    'max_grad_norm': float(os.environ.get('SM_HP_max_grad_norm', '0.5')),  # 减小梯度裁剪阈值
    'warmup_ratio': float(os.environ.get('SM_HP_warmup_ratio', '0.15')),
    'gradient_accumulation_steps': int(os.environ.get('SM_HP_gradient_accumulation_steps', '2'))
}


class CustomMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history:
            metrics = state.log_history[-1]
            print(f"\nEpoch {state.epoch:.2f} completed.")
            if "eval_loss" in metrics:
                print(f"Eval Loss: {metrics['eval_loss']:.4f}")
                print(f"Eval Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
                print(f"Eval F1: {metrics.get('eval_f1', 'N/A'):.4f}")
                print(f"Eval Precision: {metrics.get('eval_precision', 'N/A'):.4f}")
                print(f"Eval Recall: {metrics.get('eval_recall', 'N/A'):.4f}")
        else:
            print(f"Epoch {state.epoch:.2f} completed. No metrics available yet.")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_and_prepare_data():
    print(f'{hyperparameters}')

    # 获取数据路径
    train_path = os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'train_processed.jsonl')
    eval_path = os.path.join(os.environ['SM_CHANNEL_EVAL'], 'eval_processed.jsonl')

    # 加载数据集
    train_dataset = load_dataset('json', data_files=train_path, split='train')
    eval_dataset = load_dataset('json', data_files=eval_path, split='train')

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters['model_name'])

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples['text'],
            padding='max_length',
            max_length=hyperparameters['max_length'],
            truncation=True
        )

        if 'label' in examples:
            result['labels'] = examples['label']
        elif 'labels' in examples:
            result['labels'] = examples['labels']
        else:
            raise ValueError("No 'label' or 'labels' field found in the dataset")

        return result

    # 处理数据集
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # 设置数据格式
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, eval_dataset, tokenizer

class OverfitPreventionCallback(TrainerCallback):
    def __init__(self, early_stopping_threshold=0.1):
        self.early_stopping_threshold = early_stopping_threshold
        self.best_train_loss = float('inf')
        self.best_eval_loss = float('inf')

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        train_loss = metrics.get("train_loss", float('inf'))
        eval_loss = metrics.get("eval_loss", float('inf'))
        
        # 计算训练集和验证集的损失差距
        loss_gap = abs(train_loss - eval_loss)
        
        if loss_gap > self.early_stopping_threshold:
            print(f"\nWarning: Large gap between train and eval loss: {loss_gap:.4f}")
            print("Consider increasing dropout or weight decay")
        
        # 更新最佳损失
        self.best_train_loss = min(self.best_train_loss, train_loss)
        self.best_eval_loss = min(self.best_eval_loss, eval_loss)
        
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # 加载和准备数据
    train_dataset, eval_dataset, tokenizer = load_and_prepare_data()

    # 3. 训练参数
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        num_train_epochs=hyperparameters['epochs'],
        per_device_train_batch_size=hyperparameters['train_batch_size'],
        per_device_eval_batch_size=hyperparameters['eval_batch_size'],
        learning_rate=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay'],
        warmup_ratio=hyperparameters['warmup_ratio'],  # 使用ratio替代steps
        evaluation_strategy="steps",
        eval_steps=50,                # 更频繁的评估
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        gradient_accumulation_steps=hyperparameters['gradient_accumulation_steps'],
        fp16=True,
        dataloader_num_workers=4,
        group_by_length=True,
        label_smoothing_factor=hyperparameters['label_smoothing'],
        lr_scheduler_type="cosine_with_restarts",
        max_grad_norm=hyperparameters['max_grad_norm'],
        max_steps=-1,                 # 使用epochs控制
        report_to="tensorboard"
    )
    
    config = AutoConfig.from_pretrained(hyperparameters['model_name'])
    config.hidden_dropout_prob = hyperparameters['dropout_rate']
    config.attention_probs_dropout_prob = hyperparameters['dropout_rate']
    config.layer_norm_eps = 1e-7
    config.hidden_act = 'gelu_new'
    config.num_labels = 2
    
    model = AutoModelForSequenceClassification.from_pretrained(
        hyperparameters['model_name'],
        config=config
    )
    # 为所有线性层添加dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.dropout = torch.nn.Dropout(hyperparameters['dropout_rate'])
    # 初始化trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            CustomMetricsCallback(),
            EarlyStoppingCallback(
                early_stopping_patience=hyperparameters['early_stopping_patience']
            ),
            OverfitPreventionCallback(early_stopping_threshold=0.1)  # 添加新的回调
        ]
    )

    # 开始训练
    trainer.train()

    # # 评估最终模型
    # final_metrics = trainer.evaluate()
    # print("\nFinal Evaluation Metrics:")
    # print(json.dumps(final_metrics, indent=2))

    # 保存模型和tokenizer
    # 这样两者都会保存在同一个目录下，并自动上传到S3
    trainer.save_model("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")

    # # 保存评估结果
    # with open(os.path.join("/opt/ml/model", "eval_results.json"), "w") as f:
    #     json.dump(final_metrics, f, indent=2)