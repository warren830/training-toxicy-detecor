from training.prepare import prepare_data, setup_training

if __name__ == "__main__":
    # 1. 首先准备数据
    # print("准备训练数据...")
    # train_path, eval_path = prepare_data()

    # 2. 设置并启动训练
    print("开始训练任务...")
    estimator = setup_training()
    #
    # 3. 部署模型（可选）
    print("部署模型...")
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge'
    )

    # 4. 测试预测（可选）
    test_text = "测试文本"
    prediction = predictor.predict({
        "inputs": test_text
    })
    print("预测结果:", prediction)
