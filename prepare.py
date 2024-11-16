import pandas as pd
import sagemaker
from sagemaker.huggingface import HuggingFace
import boto3
from sagemaker.session import Session


# 1. 数据处理
def prepare_data():
    # 读取正负样本
    pos_df = pd.read_csv('positive.csv')
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
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

    # 上传数据到S3
    sess = sagemaker.Session()
    train_s3 = sess.upload_data('train_processed.jsonl', key_prefix='train')
    eval_s3 = sess.upload_data('eval_processed.jsonl', key_prefix='validation')
    print(f'uploaded: {train_s3}')
    # 配置训练任务
    # 1. SageMaker估计器配置
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        transformers_version='4.36.0',
        pytorch_version='2.1.0',
        py_version='py310',
        role=role,
        hyperparameters = {
            'epochs': 6,                    # 略微增加epochs
            'train_batch_size': 16,         # 保持小batch size
            'eval_batch_size': 16,
            'learning_rate': 2e-5,          # 保持较小的学习率
            'warmup_steps': 100,            # 增加warmup步数
            'weight_decay': 0.01,
            'dropout_rate': 0.3,
            'label_smoothing': 0.1,
            'warmup_ratio': 0.15,           # 新增warmup ratio
            'gradient_accumulation_steps': 2 # 新增梯度累积
        }
    )
    # 开始训练
    huggingface_estimator.fit({
        'train': train_s3,
        'eval': eval_s3
    })

    return huggingface_estimator