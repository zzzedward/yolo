from ultralytics import YOLO

print("+++++++++++++++++++++++++++++++")
print("tensorRT")
print("==================================")
print("RGB_NDVI_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_NDVI_v23/weights/best.pt"
model = YOLO(pt_path)

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'
model.export(format="engine")  # creates 'yolo11n.engine'

RT_path = pt_path.replace(".pt", ".engine")
model = YOLO(RT_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_NDVI_v23/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)

print("==================================")
print("RGB_Polar_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_Polar_v23/weights/best.pt"
model = YOLO(pt_path)

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'
model.export(format="engine")  # creates 'yolo11n.engine'

RT_path = pt_path.replace(".pt", ".engine")
model = YOLO(RT_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_Polar_v23/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)

print("==================================")
print("RGB_polar_R_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_polar_R_v23/weights/best.pt"
model = YOLO(pt_path)

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'
model.export(format="engine")  # creates 'yolo11n.engine'

RT_path = pt_path.replace(".pt", ".engine")
model = YOLO(RT_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_polar_R/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)

print("==================================")
print("RGB_R_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_R_v23/weights/best.pt"
model = YOLO(pt_path)

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'
model.export(format="engine")  # creates 'yolo11n.engine'

RT_path = pt_path.replace(".pt", ".engine")
model = YOLO(RT_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_R_v23/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)

print("==================================")
print("RGB_polar_R_Cam5_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_polar_R_Cam5_v23/weights/best.pt"
model = YOLO(pt_path)

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'
model.export(format="engine")  # creates 'yolo11n.engine'

RT_path = pt_path.replace(".pt", ".engine")
model = YOLO(RT_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_polar_R_Cam5/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)


print("+++++++++++++++++++++++++++++++")
print("pt")
print("==================================")
print("RGB_NDVI_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_NDVI_v23/weights/best.pt"
model = YOLO(pt_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_NDVI_v23/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)

print("==================================")
print("RGB_Polar_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_Polar_v23/weights/best.pt"
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

print("==================================")
print("RGB_polar_R_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_polar_R_v23/weights/best.pt"
model = YOLO(pt_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_polar_R/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)

print("==================================")
print("RGB_R_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_R_v23/weights/best.pt"
model = YOLO(pt_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_R_v23/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)

print("==================================")
print("RGB_polar_R_Cam5_v23")
# Load the YOLO11 model
pt_path = "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_polar_R_Cam5_v23/weights/best.pt"
model = YOLO(pt_path)

metrics = model.val(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/RGB_polar_R_Cam5/midfusion.yaml",
    device=0,
    batch=1,
    imgsz=640,
    amp=False,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect_val",
    name="Cam0_original_v2",
)
