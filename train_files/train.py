import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.insert(0, "/mnt/nfs_200T/optics/SHL/ultralytics")
from ultralytics import YOLO

# midfusion MoE
model = YOLO("/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/models/11/yolo11_midfusion.yaml")
model.train(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_polar_R/midfusion.yaml",
    epochs=300,
    device=[3],
    amp=False,
    batch=64,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect",
    name="Cam0_polar_R_v23"
)
model.val()