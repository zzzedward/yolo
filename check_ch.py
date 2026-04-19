# import os
# import cv2
# from glob import glob
# from multiprocessing import Pool, cpu_count
# import numpy as np

# root = "/mnt/nfs_200T/optics/data/datasetv2crop640_8ch20251201/yolo"
# cam_list = ['AoLP', 'DoLP', 'NDVI']

# expected_channels = {"Cam0": 3}
# for cam in cam_list:
#     if cam not in expected_channels:
#         expected_channels[cam] = 1


# def check_and_fix(args):
#     img_path, cam_id, exp_ch = args
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

#     if img is None:
#         return img_path, "load_fail"

#     # 判断实际通道数
#     if len(img.shape) == 2:
#         ch = 1
#     else:
#         ch = img.shape[2]

#     # Cam0（RGB）
#     if cam_id == "Cam0":
#         if ch == 3:
#             return img_path, False

#         # 单通道 → 复制为 3 通道
#         if ch == 1:
#             fixed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#             cv2.imwrite(img_path, fixed)
#             return img_path, False

#         # 多通道 → 截取前 3 通道
#         if ch > 3:
#             fixed = img[:, :, :3]
#             cv2.imwrite(img_path, fixed)
#             return img_path, False

#         return img_path, "cam0_invalid"

#     # 其他（AoLP / DoLP / NDVI）期望 1 通道
#     else:
#         # 正常：单通道
#         if ch == 1:
#             return img_path, False

#         # 3 通道 → 只能在三个通道完全相同时才能转灰度
#         if ch == 3:
#             # 判断三通道是否完全相同
#             if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
#                 fixed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 cv2.imwrite(img_path, fixed)
#                 return img_path, False
#             else:
#                 return img_path, "3ch_not_equal"

#         # 其他通道数 → 异常
#         return img_path, f"invalid_ch_{ch}"


# for cam_id in cam_list:
#     cam_dir = os.path.join(root, cam_id, "images", "val")
#     img_list = sorted(glob(os.path.join(cam_dir, "*.*")))
#     total = len(img_list)

#     print(f"\n--- 检查并修复 {cam_id} ({total} 张) ---")

#     if total == 0:
#         print("无图片")
#         continue

#     exp = expected_channels[cam_id]

#     with Pool(processes=cpu_count()) as pool:
#         results = pool.map(check_and_fix, [(p, cam_id, exp) for p in img_list])

#     # 筛选异常项
#     error_files = [(os.path.basename(p), e) for p, e in results if e]

#     print(f"异常数量 = {len(error_files)}")
#     if error_files:
#         print("示例:", error_files[:10])


import os
from glob import glob

import cv2

root = "/mnt/nfs_200T/optics/data/datasetv2crop640_8ch20251201/yolo"

cam_list = ["Cam0", "DoLP", "AoLP", "NDVI"]

for cam_id in cam_list:
    img_dir = os.path.join(root, cam_id, "images", "val")
    img_list = sorted(glob(os.path.join(img_dir, "*.*")))

    print(f"\n==== Processing {cam_id}, total {len(img_list)} images ====")

    for img_path in img_list:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"❌ Failed to read: {img_path}")
            continue

        # 判断通道
        if len(img.shape) == 2:
            # 单通道，跳过
            print(f"[Skip] Single-channel already: {os.path.basename(img_path)}")
            continue

        h, w, c = img.shape

        if c != 3:
            print(f"[Skip] Not 3-channel: {img_path}, shape={img.shape}")
            continue

        gray = img[:, :, 0]

        # 覆盖保存
        success = cv2.imwrite(img_path, gray)

        if success:
            print(f"[OK] {os.path.basename(img_path)} → Saved 1-channel")
        else:
            print(f"[Fail] Could not save: {img_path}")

print("\n=== All Done ===")
