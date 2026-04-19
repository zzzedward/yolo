import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

from ultralytics import YOLO
import glob
import numpy as np
import cv2
import time
# -------------------------------
# 配置
# -------------------------------
model_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/8_channels_clip/weights/best.pt"
channel_list = ["color", "pianzhen", "Cam3", "Cam5"]
root_dir = "/mnt/nfs_200T/optics/SHL/yolov9-main/data/test_data/clip_data"
batch_size = 16

# -------------------------------
# 加载模型
# -------------------------------
model = YOLO(model_path)

# -------------------------------
# 准备图片列表
# -------------------------------
img_lists = []
for ch in channel_list:
    ch_dir = os.path.join(root_dir, ch, "images/test")
    img_paths = sorted(glob.glob(os.path.join(ch_dir, "*.jpg")) +
                       glob.glob(os.path.join(ch_dir, "*.png")) +
                       glob.glob(os.path.join(ch_dir, "*.jpeg")))
    img_lists.append(img_paths)

# 确保每个通道文件数量一致
assert all([len(img_lists[0]) == len(img) for img in img_lists]), "各通道图片数量不一致"

# 标签路径（只用 color 通道）
label_dir = os.path.join(root_dir, "color/labels/test")

# -------------------------------
# 工具函数
# -------------------------------
def read_and_stack(idx):
    """读取每个通道同名图片并在通道维度拼接，保留原通道数"""
    imgs = []
    for ch_idx, ch in enumerate(channel_list):
        img_path = img_lists[ch_idx][idx]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保留原始通道
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # 如果是单通道，shape=(H,W)，扩展成 (H,W,1)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        imgs.append(img)
    return np.concatenate(imgs, axis=2)  # shape=(H,W,sum(通道数))

def yolo_to_xyxy(box, img_w, img_h):
    """YOLO格式(x_center, y_center, w, h) -> (x1, y1, x2, y2)"""
    x_c, y_c, w, h = box
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return [x1, y1, x2, y2]

def compute_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2-x1) * max(0, y2-y1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

# -------------------------------
# 批量推理
# -------------------------------
all_results = []
iou_list = []
non_iou_time = 0.0

for i in range(0, len(img_lists[0]), batch_size):
    t0 = time.time()
    batch_imgs = [read_and_stack(j) for j in range(i, min(i+batch_size, len(img_lists[0])))]
    
    # 推理
    results = model.predict(
        batch_imgs,
        conf=0.5,
        save=True,
        stream=False,
        verbose=False
    ) 
    all_results.extend(results)

    t1 = time.time()
    non_iou_time += (t1 - t0) 

    # 计算 IoU
    for r, idx in zip(results, range(i, min(i+batch_size, len(img_lists[0])))):
        img_h, img_w = r.orig_shape[:2]

        # 读取真实标签
        img_name = os.path.basename(img_lists[0][idx])
        label_path = os.path.join(label_dir, img_name.rsplit(".",1)[0]+".txt")
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x_c, y_c, w, h = map(float, parts[1:5])
                        gt_boxes.append(yolo_to_xyxy([x_c, y_c, w, h], img_w, img_h))

        pred_boxes = r.boxes.xyxy.cpu().numpy()
        for pbox in pred_boxes:
            if gt_boxes:
                iou_list.append(max([compute_iou(pbox, gbox) for gbox in gt_boxes]))
            else:
                iou_list.append(0.0)

# -------------------------------
# 输出结果
# -------------------------------
num_imgs = len(img_lists[0])
inference_times = [r.speed['inference'] for r in all_results]
print(f"Average inference time: {np.mean(inference_times):.4f} ms")
print(f"Total non-IoU time (read + preprocess + inference): {non_iou_time / num_imgs:.4f} s")
print(f"Average IoU: {np.mean(iou_list):.4f}")
