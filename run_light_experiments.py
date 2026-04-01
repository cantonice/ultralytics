import os
from ultralytics import YOLO

def main():
    # ==== 训练配置参数 ====
    config_dir = "/root/ultralytics/ultralytics/cfg/models/26-lightweight"
    dataset = "/root/ultralytics/ultralytics/cfg/datasets/glass_substrate.yaml"
    weights = "/root/ultralytics/yolo26n.pt"
    epochs = 500
    batch_size = 16
    project_dir = "runs/lightweight_exps2" # 实验结果统一保存的目录

    # 待训练的轻量化模型配置文件列表
    model_yamls = [
        # "yolo26.yaml",
        # "yolo26-backbone-ds.yaml",
        # "yolo26-backbone-ds-a2.yaml",
        # "yolo26-backbone-ds-a3.yaml",
        # "yolo26-backbone-ds-a4.yaml",
        # "yolo26-head-ds-c1.yaml",
        "yolo26-full-ds.yaml"
    ]

    for model_yaml in model_yamls:
        yaml_path = os.path.join(config_dir, model_yaml)
        if not os.path.exists(yaml_path):
            print(f"⚠️ 警告: 未找到 {yaml_path}，跳过该模型。")
            continue

        print(f"\n{'='*60}")
        print(f"🚀 开始训练模型: {model_yaml}")
        print(f"{'='*60}\n")

        # 1. 根据 yaml 构建新模型架构
        model = YOLO(yaml_path)

        # 2. 加载预训练权重（由于修改了部分结构，.load() 将只会加载大小和名称匹配的层参数）
        if weights and os.path.exists(weights):
            try:
                model.load(weights)
                print(f"✅ 成功从 {weights} 加载匹配的预训练层权重。")
            except Exception as e:
                print(f"⚠️ 警告: 尝试加载预训练权重时出错 ({e})")

        # 3. 运行训练
        run_name = model_yaml.replace('.yaml', '')
        model.train(
            data=dataset,
            epochs=epochs,
            batch=batch_size,
            project=project_dir,      # 统一存放在该项目目录下
            name=run_name,            # 每次运行的结果以该模型yaml的名称命名
            device='0',               # 若有多个 GPU 例如 '0,1' 可以修改此处
            workers=8,
            patience=100               # 超过100轮性能无提升则早停，可视情况修改或取消
        )
        print(f"✅ 模型 {model_yaml} 训练完成！结果保存在: {project_dir}/{run_name}\n")

if __name__ == "__main__":
    main()
