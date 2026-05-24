"""
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-12-17 15:55:35
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-12-17 16:15:48
FilePath: /ultralytics/ultralytics/nn/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE.
"""
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .tasks import (
    BaseModel,
    ClassificationModel,
    DetectionModel,
    MidFusionDetectionModel,
    SegmentationModel,
    attempt_load_one_weight,
    attempt_load_weights,
    guess_model_scale,
    guess_model_task,
    parse_model,
    torch_safe_load,
    yaml_model_load,
)

__all__ = (
    "BaseModel",
    "ClassificationModel",
    "DetectionModel",
    "MidFusionDetectionModel",
    "SegmentationModel",
    "attempt_load_one_weight",
    "attempt_load_weights",
    "guess_model_scale",
    "guess_model_task",
    "parse_model",
    "torch_safe_load",
    "yaml_model_load",
)
