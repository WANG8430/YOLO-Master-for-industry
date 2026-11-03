# 1. 创建并激活新环境
conda create -n yolo_master python=3.11 -y
conda activate yolo_master

# 2. 数据预处理（已完成）
# Video Frame Extraction:

python data/data_processing/1video_frame_extractor.py
# Human Detection and Masking:

python data/data_processing/2MASK_frame.py
# Dataset Merging:

python data/data_processing/3auto_merge_datasets.py

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 可选: 安装 FlashAttention 以加速训练 (需要 CUDA)
pip install flash_attn

# 5. 训练（已完成，训练好的权重文件best.pt）
# 修改 L-Dataset.yaml的数据集路径
python moe_train.py

# 6. 可视化推理界面
python script1.py --gui --weights best.pt
python script.py --gui --weights best.pt