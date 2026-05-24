from ultralytics import YOLO

print("Cam0_Polar_v23_yolov5_CFT=====================")
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_Polar_v23_yolov5/weights/best.pt"
model = YOLO(pt_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_Polar_v23/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)
