import pandas as pd
import sagemaker
from sagemaker.huggingface import HuggingFace
import boto3
from sagemaker.session import Session


# 1. 数据处理
def prepare_data():
    # 读取正负样本
    pos_df = pd.read_csv('positive_origin.csv')
    neg_df = pd.read_csv('negative.csv')

    # 合并数据集并准备训练格式
    train_data = pd.concat([
        pd.DataFrame({
            'text': pos_df['A'],
            'label': 1
        }),
        pd.DataFrame({
            'text': neg_df['文本'],
            'label': 0
        })
    ]).sample(frac=1).reset_index(drop=True)  # 打乱数据

    # 分割训练集和验证集
    train_size = int(0.8 * len(train_data))
    train_dataset = train_data[:train_size]
    eval_dataset = train_data[train_size:]

    # 保存为jsonl格式
    train_dataset.to_json('train.jsonl', orient='records', lines=True)
    eval_dataset.to_json('eval.jsonl', orient='records', lines=True)

    return 'train.jsonl', 'eval.jsonl'


# 2. 设置训练任务
def setup_training():
    role = 'arn:aws:iam::034362076319:role/service-role/AmazonSageMaker-ExecutionRole-20241025T115803'

    # 上传数据到S3
    sess = sagemaker.Session()
    train_s3 = sess.upload_data('train_processed.jsonl', key_prefix='train')
    eval_s3 = sess.upload_data('eval_processed.jsonl', key_prefix='validation')

    # 配置训练任务
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        requirements_file='requirements.txt',  # 添加这一行
        instance_count=1,
        role=role,
        transformers_version='4.36.0',
        pytorch_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 5,
            'train_batch_size': 64,  # 增加到64
            'eval_batch_size': 64,  # 增加到64
            'learning_rate': 3e-5,
            'model_name': 'thu-coai/roberta-base-cold',
            'warmup_ratio': 0.1,  # 改用ratio替代固定步数
            'weight_decay': 0.01,
        }
    )
    print(f'sdf')
    # 开始训练
    huggingface_estimator.fit({
        'train': train_s3,
        'eval': eval_s3
    })

    return huggingface_estimator