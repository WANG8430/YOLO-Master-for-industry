import numpy as np

# 针对 NumPy 2.0 移除 np.string_ 的补丁
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_
import os
import json
import random
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

# ==========================================
# 1. 配置区域
# ==========================================
ORIGINAL_JSON = "data/train_dataset_balanced.json"
IMG_DIR = "data/hardhat_data/train/images"
OUTPUT_JSON = "data/train_dataset_augmented.json"
MODEL_PATH = "/root/autodl-tmp/model_cache/models/Qwen/Qwen2-VL-7B-Instruct"
LORA_OUTPUT_DIR = "output_7b_augmented"


# ==========================================
# 2. 阶段 1：数据增强 (复现 MonitorVLM Dataset II 策略)
# ==========================================
def run_data_augmentation():
    print("\n" + "=" * 50)
    print(" [阶段 1] 执行数据增强 (低光+遮挡) 并构建 JSON...")

    if not os.path.exists(ORIGINAL_JSON):
        print(f" 错误：找不到原始数据集 {ORIGINAL_JSON}")
        return

    with open(ORIGINAL_JSON, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    augmented_data = []
    for item in tqdm(original_data, desc="处理进度"):
        augmented_data.append(item)  # 保留原样本
        img_path = item["image"]
        img_name = os.path.basename(img_path)

        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert("RGB")

        # 策略 A: 低光合成 (亮度降低 50-70%，模拟矿井/夜间)
        enhancer = ImageEnhance.Brightness(img)
        low_light_img = enhancer.enhance(random.uniform(0.3, 0.5))
        low_path = os.path.join(IMG_DIR, f"lowlight_{img_name}")
        low_light_img.save(low_path)

        # 策略 B: 掩码遮挡 (随机涂黑 20% 区域，模拟视角遮挡)
        img_np = np.array(img)
        h, w, _ = img_np.shape
        m_w, m_h = int(w * 0.2), int(h * 0.2)
        x, y = random.randint(0, w - m_w), random.randint(0, h - m_h)
        img_np[y:y + m_h, x:x + m_w] = 0
        mask_path = os.path.join(IMG_DIR, f"masked_{img_name}")
        Image.fromarray(img_np).save(mask_path)

        # 注入增强数据索引 [cite: 294]
        for prefix, path in [("low_", low_path), ("mask_", mask_path)]:
            new_item = item.copy()
            new_item["id"] = prefix + item["id"]
            new_item["image"] = path
            augmented_data.append(new_item)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)

    print(f" 数据准备完毕 样本数增加到 {len(augmented_data)}")


# ==========================================
# 3. 阶段 2：修正后的命令行启动 (适配 SWIFT v3.0 显式指定模块)
# ==========================================
def run_lora_training():
    # ... 之前的代码 ...
    shell_cmd = (
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "  # 减少显存碎片
        f"swift sft "
        f"--model {MODEL_PATH} "
        f"--dataset {OUTPUT_JSON} "
        f"--train_type lora "
        f"--target_modules all-linear "
        f"--lora_rank 16 "
        f"--lora_alpha 32 "
        f"--learning_rate 1e-4 "
        f"--num_train_epochs 1 "
        f"--per_device_train_batch_size 1 "  
        f"--gradient_accumulation_steps 16 "   
        f"--max_pixels 602112 "  #  限制图片分辨率，省显存
        f"--output_dir {LORA_OUTPUT_DIR} "
        f"--quant_bits 4 "
        f"--report_to none "
    )

    print(f"执行指令: {shell_cmd}")
    os.system(shell_cmd)

if __name__ == "__main__":
    run_data_augmentation()
    run_lora_training()