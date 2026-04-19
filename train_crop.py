import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from ultralytics import YOLO

model = YOLO("/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/models/11/yolo11_crop.yaml")

model.train(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/datasets/cropdata.yaml",
    epochs=300,
    imgsz=640,
    batch=64,
    device=[0,1,2,3,4,5,6,7],
    amp=False,
    workers=0,
    cache=False,
    plots=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect",
    name="cropv2"
)
# model.val()
