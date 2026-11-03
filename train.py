from ultralytics import YOLO
6


if __name__ == '__main__':
    # 加载模型
    model = YOLO('cfg/models/master/v0/det/yolo-master-n.yaml').load('YOLO-Master-EsMoE-N.pt')# 从 YAML 构建新模型

    # 训练模型
    results = model.train(
        data='L-Dataset.yaml',
        epochs=200, 
        batch=16, 
        workers=2,
        imgsz=640,
        device="0", # 使用单 GPU
        scale=0.5, 
        mosaic=1.0,
        mixup=0.0, 
        copy_paste=0.1
    )   