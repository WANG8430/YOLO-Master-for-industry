# 工业园区安全检测系统
# 功能: 1) 工人未戴安全帽 (需额外训练安全帽类别) 2) 工人离起重机过近
# 当前实现: 检测工人(类别1)和起重机(类别0)，使用单目视觉测距计算距离并标注违规
#  Behavior Magnifier (BM) 增强 - 根据标签文件裁剪放大目标区域

import argparse
import math
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image  # 用于 BM 增强

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# YOLOv5 相关导入
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
    colorstr, increment_path, non_max_suppression, print_args, scale_boxes,
    strip_optimizer, xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


# ==================== Behavior Magnifier (BM) 增强函数 ====================
def apply_behavior_magnifier(image_path, output_path, target_classes=(0, 1), scale_factor=2.0):
    """
    根据 YOLO 格式标签文件，裁剪目标区域并放大后贴回原图。
    image_path: 原始图像路径
    output_path: 增强后的图像保存路径
    target_classes: 需要放大的类别元组 (crane=0, worker=1)
    scale_factor: 放大倍数（默认2倍）
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 寻找对应的标签文件（假定与图像同名，扩展名为 .txt）
    label_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
    if not os.path.exists(label_path):
        # 无标签则直接复制原图
        img.save(output_path)
        return output_path

    enhanced = img.copy()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in target_classes:
                continue

            # 归一化坐标 -> 像素坐标
            cx_norm, cy_norm, wn_norm, hn_norm = map(float, parts[1:5])
            cx, cy = int(cx_norm * w), int(cy_norm * h)
            bw, bh = int(wn_norm * w), int(hn_norm * h)
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w, cx + bw // 2)
            y2 = min(h, cy + bh // 2)
            if x2 <= x1 or y2 <= y1:
                continue

            # 裁剪并放大
            crop = img.crop((x1, y1, x2, y2))
            new_w = int((x2 - x1) * scale_factor)
            new_h = int((y2 - y1) * scale_factor)
            crop_resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 重新计算粘贴位置（以原中心对齐）
            paste_x = max(0, cx - new_w // 2)
            paste_y = max(0, cy - new_h // 2)
            # 确保不超出边界
            if paste_x + new_w > w:
                paste_x = w - new_w
            if paste_y + new_h > h:
                paste_y = h - new_h
            enhanced.paste(crop_resized, (paste_x, paste_y))

    enhanced.save(output_path)
    return output_path


# ==================== 单目视觉测距辅助函数 ====================
def calculate_mono_distance(bbox_height_px, object_type, focal_length_px):
    """
    根据物体像素高度和真实高度估算距离（米）
    object_type: 'worker' 或 'crane'
    """
    # 固定真实高度
    HEIGHTS = {'worker': 1.7, 'crane': 4.0}
    real_height_m = HEIGHTS.get(object_type, 1.7)
    try:
        bbox_height_px = float(bbox_height_px)
        focal_length_px = float(focal_length_px)
        if bbox_height_px <= 0 or real_height_m <= 0 or focal_length_px <= 0:
            return None
        # 距离 = (焦距 * 真实高度) / 像素高度
        return 3.0 * (focal_length_px * real_height_m) / bbox_height_px
    except Exception:
        return None


def get_world_xy(pixel_coord, depth, focal_length, img_center):
    """像素坐标 + 深度 → 相机坐标系下的水平面坐标 (X, Z)"""
    u, v = pixel_coord
    cx, cy = img_center
    X = (u - cx) * depth / focal_length
    Z = depth
    return X, Z


# ==================== 安全分析核心函数 ====================
def get_center_point(bbox):
    """获取边界框的中心点（备用，使用底部中心更合理）"""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def analyze_safety_simple(detections,
                          min_distance=5.0,
                          img_shape=(800, 600),
                          focal_length=0.1,
                          worker_height=1.7,
                          crane_height=4.0,
                          pixel_to_meter_ratio=0.023):
 
    workers = []
    cranes = []

    for det in detections:
        *xyxy, conf, cls_idx = det
        cls_idx = int(cls_idx)

        x1, y1, x2, y2 = map(float, xyxy)
        bbox_h = y2 - y1
        bottom_center = ((x1 + x2) / 2.0, float(y2))
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        if cls_idx == 0:  # crane
            cranes.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'bbox_h': bbox_h,
                'center': center,
                'bottom': bottom_center,
                'type': 'crane'
            })
        elif cls_idx == 1:  # worker
            workers.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'bbox_h': bbox_h,
                'center': center,
                'bottom': bottom_center,
                'type': 'worker'
            })

    safety_issues = []
    img_h, img_w = img_shape[:2]
    img_center = (img_w / 2.0, img_h / 2.0)

    use_mono = (focal_length is not None) and (focal_length > 0)


    for i, worker in enumerate(workers):
        for j, crane in enumerate(cranes):
            if use_mono:
                # 单目估算各自深度，类型自动选择高度
                dist_w = calculate_mono_distance(worker['bbox_h'], worker['type'], focal_length)
                dist_c = calculate_mono_distance(crane['bbox_h'], crane['type'], focal_length)
                if dist_w is None or dist_c is None:
                    # 回退至像素比例法
                    px_dist = math.hypot(
                        worker['bottom'][0] - crane['bottom'][0],
                        worker['bottom'][1] - crane['bottom'][1]
                    )
                    actual_distance = px_dist * pixel_to_meter_ratio
                else:
                    # 计算世界坐标并求欧氏距离
                    X_w, Z_w = get_world_xy(worker['bottom'], dist_w, focal_length, img_center)
                    X_c, Z_c = get_world_xy(crane['bottom'], dist_c, focal_length, img_center)
                    actual_distance = math.sqrt((X_w - X_c)**2 + (Z_w - Z_c)**2)
            else:
                # 像素比例法（使用底部中心点计算像素距离）
                px_dist = math.hypot(
                    worker['bottom'][0] - crane['bottom'][0],
                    worker['bottom'][1] - crane['bottom'][1]
                )
                actual_distance = px_dist * pixel_to_meter_ratio

            if actual_distance < min_distance:
                safety_issues.append({
                    'type': 'crane_too_close',
                    'message': f'工人{i+1} 距吊车{j+1}: {actual_distance:.1f}m',
                    'worker_bbox': worker['bbox'],
                    'crane_bbox': crane['bbox'],
                    'distance': actual_distance,
                    'worker_center': worker['center'],
                    'crane_center': crane['center']
                })

    stats = {
        'total_workers': len(workers),
        'total_cranes': len(cranes),
        'crane_violations': len(safety_issues),
        'total_violations': len(safety_issues)
    }
    safety_score = max(0, 100 - len(safety_issues) * 10)
    return safety_issues, stats, safety_score


def draw_safety_info(image, stats, safety_score, safety_issues):
    """在图像上绘制安全状态栏、统计信息和违规连线"""
    h, w = image.shape[:2]

    # 状态条
    if safety_score >= 80:
        color = (0, 255, 0)
        status = "SAFE"
    elif safety_score >= 60:
        color = (0, 255, 255)
        status = "CAUTION"
    else:
        color = (0, 0, 255)
        status = "DANGER"
    cv2.rectangle(image, (0, 0), (w, 50), color, -1)
    cv2.putText(image, f"Status: {status} ({int(safety_score)})", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # 统计信息（右上角）
    stats_text = [
        f"Workers: {stats['total_workers']}",
        f"Cranes: {stats['total_cranes']}",
        f"Violations: {stats['crane_violations']}"
    ]
    y_offset = 90
    for text in stats_text:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(image, (w - text_size[0] - 30, y_offset - 30),
                      (w - 10, y_offset + 10), (0, 0, 0), -1)
        cv2.putText(image, text, (w - text_size[0] - 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 40

    # 绘制违规连线
    for issue in safety_issues:
        # 中心点已经是 Python float，转为 int 供 OpenCV 使用
        worker_center = (int(issue['worker_center'][0]), int(issue['worker_center'][1]))
        crane_center = (int(issue['crane_center'][0]), int(issue['crane_center'][1]))

        cv2.line(image, worker_center, crane_center, (0, 165, 255), 3)
        cv2.circle(image, worker_center, 6, (255, 0, 0), -1)
        cv2.circle(image, crane_center, 6, (0, 0, 255), -1)
        mid_x = (worker_center[0] + crane_center[0]) // 2
        mid_y = (worker_center[1] + crane_center[1]) // 2
        distance_text = f"{issue['distance']:.1f}m"
        text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (mid_x - text_size[0] // 2 - 10, mid_y - text_size[1] // 2 - 10),
                      (mid_x + text_size[0] // 2 + 10, mid_y + text_size[1] // 2 + 10),
                      (0, 0, 0), -1)
        cv2.putText(image, distance_text,
                    (mid_x - text_size[0] // 2, mid_y + text_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # 标记违规框
        x1, y1, x2, y2 = map(int, issue['worker_bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        x1, y1, x2, y2 = map(int, issue['crane_bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


# ==================== 命令行推理模式 ====================
@smart_inference_mode()
def run(
        weights=ROOT / "best.pt",
        source=ROOT / "data/images",
        data=ROOT / "data/coco128.yaml",
        imgsz=(800, 600),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device="",
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / "runs/detect",
        name="exp",
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
        min_crane_distance=5.0,
        apply_bm=False,
        focal_length=0.1,
        worker_height=1.7,
        crane_height=4.0,
        pixel_to_meter_ratio=0.023,
):
    # 确保所有路径参数为字符串，避免类型错误
    source = str(source)
    weights = [str(w) for w in weights] if isinstance(weights, list) else str(weights)
    data = str(data) if data else data
    project = str(project)

    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        if apply_bm and not webcam and not screenshot and dataset.mode == "image":
            temp_path = str(save_dir / f"_temp_bm_{Path(path).stem}.jpg")
            apply_behavior_magnifier(path, temp_path, target_classes=(0, 1), scale_factor=2.0)
            temp_dataset = LoadImages(temp_path, img_size=imgsz, stride=stride, auto=pt)
            _, im_temp, im0s_temp, _, _ = next(iter(temp_dataset))
            im, im0s = im_temp, im0s_temp

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_path)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "%gx%g " % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0

            safety_issues = []
            stats = {}
            safety_score = 0

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                safety_issues, stats, safety_score = analyze_safety_simple(
                    det,
                    min_distance=min_crane_distance,
                    img_shape=im0.shape,
                    focal_length=focal_length,
                    worker_height=worker_height,
                    crane_height=crane_height,
                    pixel_to_meter_ratio=pixel_to_meter_ratio
                )

                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f"{names.get(c, str(c))} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

                im0 = draw_safety_info(im0, stats, safety_score, safety_issues)

                if save_txt:
                    with open(f"{txt_path}.txt", "a") as f:
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                if save_crop:
                    for *xyxy, conf, cls in reversed(det):
                        cls_name = names[int(cls)] if isinstance(names, dict) else str(cls)
                        save_one_box(xyxy, imc, file=save_dir / "crops" / cls_name / f"{p.stem}.jpg", BGR=True)
            else:
                stats = {'total_workers': 0, 'total_cranes': 0, 'crane_violations': 0, 'total_violations': 0}
                safety_score = 100
                im0 = draw_safety_info(im0, stats, safety_score, [])

            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


# ==================== GUI 界面 ====================
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import ImageTk
import threading

class SafetyDetectorGUI:
    def __init__(self, opt):
        self.opt = opt
        self.model = None
        self.device = None
        self.names = None
        self.stride = None
        self.pt = None

        self.root = tk.Tk()
        self.root.title("工业园区安全检测系统")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')

        self.title_font = ('Microsoft YaHei', 18, 'bold')
        self.label_font = ('Microsoft YaHei', 12)
        self.stats_font = ('Consolas', 15)
        self.violation_font = ('Consolas', 13)

        control_frame = tk.Frame(self.root, bg='#34495e', height=80)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        control_frame.pack_propagate(False)

        btn_style = {'font': self.label_font, 'bg': '#1abc9c', 'fg': 'white',
                     'activebackground': '#16a085', 'activeforeground': 'white',
                     'relief': tk.FLAT, 'padx': 15, 'pady': 8, 'bd': 0}

        self.btn_load = tk.Button(control_frame, text="📂 加载图片", command=self.load_image, **btn_style)
        self.btn_load.pack(side=tk.LEFT, padx=10)
        self.btn_detect = tk.Button(control_frame, text="🔍 开始检测", command=self.start_detect,
                                    state=tk.DISABLED, **btn_style)
        self.btn_detect.pack(side=tk.LEFT, padx=10)
        self.lbl_path = tk.Label(control_frame, text="未选择图片", fg="#ecf0f1", bg='#34495e',
                                 font=self.label_font, anchor='w')
        self.lbl_path.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas_frame = tk.Frame(main_frame, bg='#2c3e50')
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg='#1a1a2e', highlightthickness=0)
        self.scroll_y = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_frame = tk.Frame(main_frame, width=350, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        info_frame.pack_propagate(False)

        tk.Label(info_frame, text="📊 检测统计", font=self.title_font, bg='#ecf0f1', fg='#2c3e50').pack(pady=15)
        stats_bg = tk.Frame(info_frame, bg='white', relief=tk.SUNKEN, bd=1)
        stats_bg.pack(pady=5, padx=10, fill=tk.X)
        self.stats_text = tk.Text(stats_bg, height=8, width=35, font=self.stats_font,
                                  bg='white', fg='#2c3e50', bd=0, highlightthickness=0)
        self.stats_text.pack(padx=5, pady=5)

        tk.Label(info_frame, text="⚠️ 违规详情", font=self.title_font, bg='#ecf0f1', fg='#c0392b').pack(pady=10)
        violation_bg = tk.Frame(info_frame, bg='white', relief=tk.SUNKEN, bd=1)
        violation_bg.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        self.violations_list = scrolledtext.ScrolledText(violation_bg, height=15, width=35,
                                                          font=self.violation_font,
                                                          bg='white', fg='#2c3e50', bd=0, highlightthickness=0)
        self.violations_list.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar()
        self.status_var.set("✅ 就绪 | 未加载模型")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN,
                              anchor=tk.W, font=('Microsoft YaHei', 10), bg='#bdc3c7', fg='#2c3e50')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.after(100, self.init_model)

    def init_model(self):
        def load():
            self.update_status("🔄 正在加载模型，请稍候...")
            try:
                self.device = select_device(self.opt.device)
                self.model = DetectMultiBackend(str(self.opt.weights[0]), device=self.device,
                                                dnn=self.opt.dnn, data=str(self.opt.data),
                                                fp16=self.opt.half)
                self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
                self.update_status("✅ 模型加载完成 | 就绪")
            except Exception as e:
                self.update_status(f"❌ 模型加载失败: {str(e)}")
                messagebox.showerror("错误", f"模型加载失败\n{str(e)}")
        threading.Thread(target=load, daemon=True).start()

    def update_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        if file_path:
            self.image_path = file_path
            self.lbl_path.config(text=f"📁 {file_path}", fg="#1abc9c")
            self.btn_detect.config(state=tk.NORMAL)
            self.display_image(cv2.imread(file_path))

    def display_image(self, cv_img):
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 50 else 700
        canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 50 else 600
        pil_img.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def start_detect(self):
        if not hasattr(self, 'image_path'):
            messagebox.showwarning("警告", "请先加载图片")
            return
        if self.model is None:
            messagebox.showwarning("警告", "模型尚未加载完成，请稍候")
            return
        self.btn_detect.config(state=tk.DISABLED)
        self.update_status("🔎 正在检测...")
        threading.Thread(target=self.detect_image, daemon=True).start()

    def detect_image(self):
        try:
            from utils.general import non_max_suppression, scale_boxes
            from utils.dataloaders import LoadImages

            dataset = LoadImages(self.image_path, img_size=self.opt.imgsz, stride=self.stride, auto=self.pt)
            path, im, im0s, vid_cap, s = next(iter(dataset))

            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

            pred = self.model(im, augment=self.opt.augment, visualize=False)
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                       self.opt.classes, self.opt.agnostic_nms, max_det=self.opt.max_det)

            det = pred[0]
            im0 = im0s.copy()
            safety_issues, stats, safety_score = [], {}, 0
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                safety_issues, stats, safety_score = analyze_safety_simple(
                    det,
                    min_distance=self.opt.min_crane_distance,
                    img_shape=im0.shape,
                    focal_length=self.opt.focal_length,
                    worker_height=self.opt.worker_height,
                    crane_height=self.opt.crane_height,
                    pixel_to_meter_ratio=self.opt.pixel_to_meter_ratio
                )
                im0 = draw_safety_info(im0, stats, safety_score, safety_issues)

            self.root.after(0, self.update_result, im0, stats, safety_score, safety_issues)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("检测错误", str(e)))
            self.root.after(0, lambda: self.btn_detect.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.update_status("❌ 检测失败"))

    def update_result(self, result_img, stats, safety_score, safety_issues):
        self.display_image(result_img)
        stats_str = f"👷  工人数量: {stats.get('total_workers', 0)}\n"
        stats_str += f"🏗️吊车数量: {stats.get('total_cranes', 0)}\n"
        stats_str += f"⚠️  违规数量: {stats.get('crane_violations', 0)}\n"
        stats_str += f"📊  安全评分: {safety_score}/100"
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_str)

        self.violations_list.delete(1.0, tk.END)
        if safety_issues:
            for idx, issue in enumerate(safety_issues, 1):
                self.violations_list.insert(tk.END, f"{idx}. {issue['message']}\n")
        else:
            self.violations_list.insert(tk.END, "✅ 无违规行为")

        self.btn_detect.config(state=tk.NORMAL)
        self.update_status("✅ 检测完成")

    def run(self):
        self.root.mainloop()


# ==================== 主入口 ====================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "weights/map_93_best.pt",
                        help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images",
                        help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/classes6_DA.yaml",
                        help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640],
                        help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--min-crane-distance", type=float, default=5.0,
                        help="起重机最小安全距离（米）")
    parser.add_argument("--apply-bm", action="store_true", help="使用 Behavior Magnifier 增强图像")
    parser.add_argument("--gui", action="store_true", help="启动图形化界面")
    parser.add_argument("--focal-length", type=float, default=None,
                        help="相机焦距（像素），用于单目测距，若不提供则使用像素比例法")
    parser.add_argument("--worker-height", type=float, default=1.7,
                        help="工人真实高度（米）")
    parser.add_argument("--crane-height", type=float, default=3.0,
                        help="起重机真实高度（米）")
    parser.add_argument("--pixel-to-meter-ratio", type=float, default=0.058,
                        help="像素比例法换算因子（回退时使用）")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    # 确保所有路径为字符串，杜绝 Path 对象引起的问题
    opt.weights = [str(Path(w).resolve()) for w in opt.weights]
    opt.source = str(opt.source)
    if opt.data:
        opt.data = str(opt.data)
    opt.project = str(opt.project)

    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    if opt.gui:
        app = SafetyDetectorGUI(opt)
        app.run()
    else:
        opt_dict = vars(opt).copy()
        opt_dict.pop('gui', None)
        run(**opt_dict)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)