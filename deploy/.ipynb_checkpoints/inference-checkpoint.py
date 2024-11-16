import os
import json
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def model_fn(model_dir,context=None):
    """加载模型和分词器"""
    logger.info(f"Loading model from {model_dir}")
    try:
        # # 如果是从S3加载的模型
        # if os.path.exists(os.path.join(model_dir, "model.tar.gz")):
        #     model_path = model_dir
        # else:
        #     # 如果是从Hugging Face Hub加载的模型
        #     print('error when loading models')
            
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        if torch.cuda.is_available():
            logger.info("Moving model to GPU")
            model = model.cuda()
            
        model.eval()
        logger.info("Model loading completed")
        # 返回字典而不是元组
        return {
            "model": model,
            "tokenizer": tokenizer
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """处理输入请求"""
    logger.info(f"Received request with content type: {request_content_type}")
    try:
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            logger.info(f"Processed input: {input_data}")
            return input_data
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        logger.error(f"Error parsing input: {str(e)}")
        raise

def predict_fn(input_data, model_dict, context=None):
    """执行预测"""
    logger.info("Starting prediction")
    try:
        # 从字典中获取模型和tokenizer
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        text = input_data.get('text', '')
        
        logger.info(f"Processing text: {text}")
        
        # tokenize输入文本
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # 如果模型在GPU上，将输入也移到GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 执行预测
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取预测结果
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
        
        result = {
            'prediction': prediction.cpu().item(),
            'probabilities': probabilities.cpu().tolist()[0]
        }
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def output_fn(prediction, response_content_type):
    """格式化输出结果"""
    logger.info(f"Formatting output with content type: {response_content_type}")
    try:
        if response_content_type == 'application/json':
            json_output = json.dumps(prediction)
            logger.info(f"Formatted output: {json_output}")
            return json_output
        else:
            raise ValueError(f"Unsupported content type: {response_content_type}")
    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        raise