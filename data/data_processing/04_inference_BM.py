import json
import torch
import os
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp/model_cache"
from modelscope import snapshot_download
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import PeftModel


# ==========================================
# 1. Oracle 版 Behavior Magnifier (BM) 模块
# (直接利用数据集的真实标签进行精准裁剪和放大)
# ==========================================
def apply_behavior_magnifier(image_path, output_path):
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # 自动寻找对应的标签文件
    label_path = image_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
    enhanced_img = img.copy()

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(parts[0])

                # 0=helmet, 1=person，将人和安全帽都放大
                if cls_id in [0, 1]:
                    cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])

                    # 还原为真实像素坐标
                    cx, cy = int(cx_norm * img_w), int(cy_norm * img_h)
                    bw, bh = int(w_norm * img_w), int(h_norm * img_h)

                    x1 = max(0, cx - bw // 2)
                    y1 = max(0, cy - bh // 2)
                    x2 = min(img_w, cx + bw // 2)
                    y2 = min(img_h, cy + bh // 2)

                    if x2 <= x1 or y2 <= y1: continue

                    # 裁剪并放大 2 倍
                    crop = img.crop((x1, y1, x2, y2))
                    new_w, new_h = (x2 - x1) * 2, (y2 - y1) * 2
                    crop_resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

                    # 无缝重插入
                    paste_x1 = max(0, cx - new_w // 2)
                    paste_y1 = max(0, cy - new_h // 2)
                    enhanced_img.paste(crop_resized, (paste_x1, paste_y1))

    enhanced_img.save(output_path)
    return output_path


# ==========================================
# 2. VLM 模型推理
# ==========================================
def run_vlm_inference(model, processor, image_path, prompt):
    messages = [
        {"role": "user",
         "content": [{"type": "image", "image": image_path, "max_pixels": 313600}, {"type": "text", "text": prompt}]}
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True,
                       return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.1)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = \
        processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text


# ==========================================
# 3. 主程序：批量测评与算分
# ==========================================
if __name__ == "__main__":
    print("正在查找本地模型缓存...")
    model_name = snapshot_download("Qwen/Qwen2-VL-7B-Instruct")
    lora_path = "output_7b/v2-20260303-003624/checkpoint-75"

    with open("data/test_dataset_real.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print("加载 Qwen2-VL-7B (4-bit)...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="auto",
                                                            quantization_config=quantization_config)
    processor = AutoProcessor.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, lora_path)

    y_true = []
    y_pred = []

    print(f"\n🚀 开始执行 Oracle BM 增强版批量推理 ({len(test_data)} 张图片)，请耐心等待...")
    prompt_text = "请仔细检查这张工地监控截图，根据《建筑施工安全检查标准》，输出结构化的安全隐患识别报告。"

    # 临时文件路径，每次循环覆盖
    temp_enhanced_path = "bm_temp.jpg"

    for item in tqdm(test_data):
        img_path = item["image"]
        ground_truth = item["ground_truth_hazard"]
        y_true.append(1 if ground_truth else 0)

        # 步骤 A：应用 BM 放大，生成增强图像
        apply_behavior_magnifier(img_path, temp_enhanced_path)

        # 步骤 B：VLM 对增强图像进行推理
        output_text = run_vlm_inference(model, processor, temp_enhanced_path, prompt_text)

        # 步骤 C：自动阅卷逻辑
        output_str = output_text.replace(" ", "").replace("\n", "").replace("'", "\"")
        if "\"存在隐患\":true" in output_str:
            y_pred.append(1)
        else:
            y_pred.append(0)

    # 4. 计算并打印评估指标
    print("\n" + "=" * 50)
    print("🏆 终极指标：加入 Oracle BM 行为放大后的评估报告")
    print("=" * 50)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"F1-Score          : {f1:.4f} (与之前 0.6349 相比，是否大幅提升？)")
    print("\n混淆矩阵 (Confusion Matrix):")
    print(f"                 预测: 合规(0)  预测: 违规(1)")
    print(f"实际: 合规(0)      {cm[0][0]:<12} {cm[0][1]}")
    if len(cm) > 1:
        print(f"实际: 违规(1)      {cm[1][0]:<12} {cm[1][1]}")
    print("=" * 50)