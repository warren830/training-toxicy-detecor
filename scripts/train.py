import os
import random

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 定义hyperparameters，添加默认值
hyperparameters = {
    'learning_rate': float(os.environ.get('SM_HP_learning_rate', '3e-5')),
    'train_batch_size': int(os.environ.get('SM_HP_train_batch_size', '32')),
    'eval_batch_size': int(os.environ.get('SM_HP_eval_batch_size', '32')),
    'epochs': float(os.environ.get('SM_HP_epochs', '5')),
    'weight_decay': float(os.environ.get('SM_HP_weight_decay', '0.01')),
    'warmup_steps': int(os.environ.get('SM_HP_warmup_steps', '500')),
    'model_name': os.environ.get('SM_HP_model_name', 'thu-coai/roberta-base-cold'),
    'max_length': int(os.environ.get('SM_HP_max_length', '128')),
    'warmup_ratio': float(os.environ.get('SM_HP_warmup_ratio', '0.1')),
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


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        hyperparameters['model_name'],
        num_labels=2
    )


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed. Metrics: {state.log_history[-1]}")


def compute_loss(self, model, inputs, return_outputs=False):
    """
    重写计算损失函数的方法
    Args:
        model: 模型
        inputs: 输入数据
        return_outputs: 是否返回模型输出
    """
    labels = inputs.get("labels")

    # 前向传播
    outputs = model(**inputs)
    logits = outputs.get("logits")

    # 设置损失函数权重
    weights = torch.tensor([2.0]).to(self.args.device)  # 负样本权重为正样本2倍
    loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)

    # 计算损失
    loss = loss_fct(logits.view(-1), labels.float().view(-1))

    return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # 加载和准备数据
    train_dataset, eval_dataset, tokenizer = load_and_prepare_data()
    try:
        print("Starting training...")
        print(f"Hyperparameters: {hyperparameters}")

        # # 加载数据
        # print("Loading dataset...")
        # dataset = load_dataset('json', data_files={
        #     'train': '/opt/ml/input/data/train/train.jsonl',
        #     'validation': '/opt/ml/input/data/eval/eval.jsonl'
        # })
        # print(f"Dataset loaded: {dataset}")

        # # 初始化tokenizer
        # print("Initializing tokenizer...")
        # tokenizer = AutoTokenizer.from_pretrained(hyperparameters['model_name'])
        #
        # # 数据预处理
        # print("Tokenizing datasets...")
        # tokenized_datasets = dataset.map(
        #     tokenize_function,
        #     batched=True,
        #     remove_columns=dataset["train"].column_names
        # )
        # 计算相关参数
        total_train_samples = 11038  # 7361 + 3677
        steps_per_epoch = total_train_samples // hyperparameters['train_batch_size']

        # 使用hyperparameters创建training_args
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=hyperparameters['epochs'],
            per_device_train_batch_size=hyperparameters['train_batch_size'],
            per_device_eval_batch_size=hyperparameters['eval_batch_size'],
            learning_rate=hyperparameters['learning_rate'],
            warmup_ratio=hyperparameters['warmup_ratio'],
            weight_decay=hyperparameters['weight_decay'],
            evaluation_strategy="epoch",  # 改为 "epoch"
            save_strategy="epoch",  # 与 evaluation_strategy 保持一致
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir="/opt/ml/output/logs"
        )
        # 训练参数
        # print("Setting up training arguments...")
        # training_args = TrainingArguments(
        #     output_dir="/opt/ml/model",
        #     evaluation_strategy="epoch",
        #     learning_rate=hyperparameters['learning_rate'],
        #     per_device_train_batch_size=hyperparameters['train_batch_size'],
        #     per_device_eval_batch_size=hyperparameters['eval_batch_size'],
        #     num_train_epochs=hyperparameters['epochs'],
        #     weight_decay=hyperparameters['weight_decay'],
        #     warmup_steps=hyperparameters['warmup_steps'],
        #     save_strategy="epoch",
        #     load_best_model_at_end=True,
        #     metric_for_best_model="f1",
        #     save_total_limit=2,
        #     logging_dir="/opt/ml/output/logs"
        # )

        # 初始化trainer
        print("Initializing trainer...")

        # 使用CustomTrainer
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_loss=compute_loss,
            compute_metrics=compute_metrics,
            callbacks=[CustomCallback()]
        )
        # 开始训练
        print("Starting training process...")
        trainer.train()

        # 保存模型
        print("Saving model...")
        trainer.save_model()
        print("Training completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise