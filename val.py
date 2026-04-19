# from ultralytics import YOLO

# model = YOLO("/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/train/weights/best.pt")

# # Customize validation settings
# metrics = model.val(data="/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/datasets/data.yaml", imgsz=2000, batch=16, conf=0.25, iou=0.6, device="0, 1, 2, 3")

import time
from ultralytics import YOLO
import torch

# 单卡推理，假设使用 GPU 0
device_id = "0"
torch.cuda.set_device(int(device_id))

# 加载模型
model = YOLO("/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/train/weights/best.pt")

# 测试图片路径
img_path = "/mnt/nfs_200T/optics/SHL/yolov9-main/data/8_CamData/color/images/test/4_0_17.jpg"

# 开始计时
start_time = time.time()

# 推理
results = model.predict(
    source=img_path,
    imgsz=2000,   # 将输入缩放到2K分辨率（保持宽高比）
    conf=0.25,    # 置信度阈值
    iou=0.6,      # NMS阈值
    device=device_id,
    save=False,
    verbose=False
)

end_time = time.time()
inference_time = end_time - start_time

print(f"单张图片推理时间: {inference_time:.4f} s")
print("预测框 (xyxy 格式):")
for r in results:
    print(r.boxes.xyxy.cpu().numpy())
