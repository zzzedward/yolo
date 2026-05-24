from ultralytics import YOLO

# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_v23_yolo11/weights/best.pt"
model = YOLO(pt_path)

# # Export the model to ONNX format
# model.export(format="onnx")  # creates 'yolo11n.onnx'
# model.export(format="engine")  # creates 'yolo11n.engine'
# RT_path = pt_path.replace(".pt", ".engine")
# model = YOLO(RT_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/datasets/data.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)
