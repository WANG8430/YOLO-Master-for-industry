from ultralytics import YOLO


if __name__ == '__main__':
    # 加载 MoE 配置
    model = YOLO("ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml").load('YOLO-Master-EsMoE-N.pt')

    # MoE 训练
    results = model.train(
        data="L-Dataset.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        moe_num_experts=8,      # 专家数量
        moe_top_k=2,            # 每个 token 激活的专家数
        moe_balance_loss=0.01,  # 负载均衡损失权重
    )

    # 专家利用率分析与剪枝
    model.prune_experts(threshold=0.15)
