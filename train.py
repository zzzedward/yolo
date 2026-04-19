import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from ultralytics import YOLO
import torch

# midfusion
# model = YOLO("/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/models/v5/yolov5_midfusion.yaml")
model = YOLO("/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/models/11/yolo11_midfusion.yaml")
# # load pretrain
# model.load(
#     "/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect/Cam0_v23_yolo11/weights/best.pt"
# )
# sd = model.model.state_dict()

# # backbone1: layers 0–8
# # backbone2: layers 9–17
# for i in range(9):
#     for k in list(sd.keys()):
#         prefix1 = f"model.{i}."
#         prefix2 = f"model.{i + 9}."
#         if k.startswith(prefix1):
#             k2 = k.replace(prefix1, prefix2, 1)
#             if k2 in sd:
#                 sd[k2].copy_(sd[k])  # 数值复制（不是引用）

# # model.model.load_state_dict(sd, strict=False)
# p1 = model.model.model[0].conv.weight
# p2 = model.model.model[9].conv.weight

# print(torch.allclose(p1, p2))     # True
# print(p1.data_ptr() == p2.data_ptr())  # False（必须）

model.train(
    data="/mnt/nfs_200T/optics/SHL/ultralytics/train_files/FLIR/midfusion.yaml",
    epochs=300,
    device=[1],
    amp=False,
    batch=128,
    project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect",
    name="FLIR_yolov11"
)
model.val()

# # midfusion MoE
# model = YOLO("/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/models/11/MoE/yolo11_moe_midfusion_TokenLevel_test.yaml")
# model.train(
#     data="/mnt/nfs_200T/optics/SHL/ultralytics/ultralytics/cfg/datasets/midfusionMoE.yaml",
#     epochs=300,
#     device=[0],
#     amp=False,
#     batch=64,
#     project="/mnt/nfs_200T/optics/SHL/ultralytics/runs/detect",
#     name="midfusion_RGB_DoLP_NDVI_Cam4_6_tokenlevel_test_v2"
# )
# model.val()