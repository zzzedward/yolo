# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.

    Attributes:
        img_path (str): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float): Fraction of dataset to utilize.
        channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        cv2_flag (int): OpenCV flag for reading images.
        im_files (List[str]): List of image file paths.
        labels (List[Dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (List[Path]): List of numpy file paths.
        cache (str): Cache images to RAM or disk during training.
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Set the shape of bounding boxes as rectangles.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str | List[str]): Path to the folder containing images or list of image paths.
            imgsz (int): Image size for resizing.
            cache (bool | str): Cache images to RAM or disk during training.
            augment (bool): If True, data augmentation is applied.
            hyp (Dict[str, Any]): Hyperparameters to apply data augmentation.
            prefix (str): Prefix to print in log messages.
            rect (bool): If True, rectangular training is used.
            batch_size (int): Size of batches.
            stride (int): Stride used in the model.
            pad (float): Padding value.
            single_cls (bool): If True, single class training is used.
            classes (List[int], optional): List of included classes.
            fraction (float): Fraction of dataset to utilize.
            channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read image files from the specified path.

        Args:
            img_path (str | List[str]): Path or list of paths to image directories or files.

        Returns:
            (List[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """Update labels to include only specified classes.

        Args:
            include_class (List[int], optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (Tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (Tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writeable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (Dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError


class MultiChannelBaseDataset(Dataset):
    """Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.

    Attributes:
        img_path (str): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float): Fraction of dataset to utilize.
        channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        cv2_flag (int): OpenCV flag for reading images.
        im_files (List[str]): List of image file paths.
        labels (List[Dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (List[Path]): List of numpy file paths.
        cache (str): Cache images to RAM or disk during training.
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Set the shape of bounding boxes as rectangles.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str | List[str]): Path to the folder containing images or list of image paths.
            imgsz (int): Image size for resizing.
            cache (bool | str): Cache images to RAM or disk during training.
            augment (bool): If True, data augmentation is applied.
            hyp (Dict[str, Any]): Hyperparameters to apply data augmentation.
            prefix (str): Prefix to print in log messages.
            rect (bool): If True, rectangular training is used.
            batch_size (int): Size of batches.
            stride (int): Stride used in the model.
            pad (float): Padding value.
            single_cls (bool): If True, single class training is used.
            classes (List[int], optional): List of included classes.
            fraction (float): Fraction of dataset to utilize.
            channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [[None] * self.ni] * len(self.im_files), [None] * self.ni, [None] * self.ni

        self.npy_files = [[] for _ in range(len(self.im_files))]
        for i in range(len(self.im_files)):
            self.npy_files[i] = [Path(f).with_suffix(".npy") for f in self.im_files[i]]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read image files from the specified path.

        Args:
            img_path (str | List[str]): Path or list of paths to image directories or files.

        Returns:
            (List[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f, im_files = [[] for _ in range(len(img_path))], [[] for _ in range(len(img_path))]  # image files
            for i, path in enumerate(img_path):
                for p in path if isinstance(path, list) else [path]:
                    p = Path(p)  # os-agnostic
                    if p.is_dir():  # dir
                        f[i] += glob.glob(str(p / "**" / "*.*"), recursive=True)
                        # F = list(p.rglob('*.*'))  # pathlib
                    elif p.is_file():  # file
                        with open(p, encoding="utf-8") as t:
                            t = t.read().strip().splitlines()
                            parent = str(p.parent) + os.sep
                            f[i] += [
                                x.replace("./", parent) if x.startswith("./") else x for x in t
                            ]  # local to global path
                            # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                    else:
                        raise FileNotFoundError(f"{self.prefix}{p} does not exist")
                im_files[i] = sorted(
                    x.replace("/", os.sep) for x in f[i] if x.rpartition(".")[-1].lower() in IMG_FORMATS
                )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """Update labels to include only specified classes.

        Args:
            include_class (List[int], optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (Tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (Tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        image = []
        # if self.ims[0][i] is None:  # not cached in RAM
        for m in range(len(self.im_files)):
            im, f, fn = self.ims[m][i], self.im_files[m][i], self.npy_files[m][i]
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    if any(
                        x in f for x in ("NDVI", "DoLP", "AoLP", "Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6", "Cam7")
                    ):
                        im = imread(f, flags=cv2.IMREAD_GRAYSCALE)
                    else:
                        im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                if any(
                    x in f for x in ("NDVI", "DoLP", "AoLP", "Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6", "Cam7")
                ):
                    im = imread(f, flags=cv2.IMREAD_GRAYSCALE)
                else:
                    im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[m][i], self.im_hw0[i], self.im_hw[i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[m][j], self.im_hw0[j], self.im_hw[j] = None, None, None

            image.append(im)
        image = np.concatenate(image, axis=-1)
        return image, (h0, w0), im.shape[:2]
        # else:
        #     for m in range(len(self.im_files)):
        #         image.append(self.ims[m][i])
        #     image = np.concatenate(image, axis=-1)
        #     return image, self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                import pdb

                pdb.set_trace()
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writeable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        for j in range(len(self.im_files)):
            self.im_files[j] = [self.im_files[j][i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (Dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError


class CropBaseDataset(Dataset):
    """Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.

    Attributes:
        img_path (str): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float): Fraction of dataset to utilize.
        channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        cv2_flag (int): OpenCV flag for reading images.
        im_files (List[str]): List of image file paths.
        labels (List[Dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (List[Path]): List of numpy file paths.
        cache (str): Cache images to RAM or disk during training.
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Set the shape of bounding boxes as rectangles.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        crop_size=640,
        overlap=0,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.crop_size = crop_size
        self.overlap = overlap
        self.stride_crop = crop_size - overlap if crop_size > overlap else crop_size

        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)
        self.ni = len(self.labels)

        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        orig_files = getattr(self, "orig_im_files", self.im_files)
        self.ims, self.im_hw0, self.im_hw = [None] * len(orig_files), [None] * len(orig_files), [None] * len(orig_files)
        self.npy_files = [Path(f).with_suffix(".npy") for f in orig_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read image files from the specified path.

        Args:
            img_path (str | List[str]): Path or list of paths to image directories or files.

        Returns:
            (List[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """Update labels to include only specified classes.

        Args:
            include_class (List[int], optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load original image by original-image index i. 这里的 i 是原图索引，不是 crop 样本索引。.
        """
        file_list = getattr(self, "orig_im_files", self.im_files)

        if not hasattr(self, "orig_ims"):
            self.orig_ims = [None] * len(file_list)
            self.orig_im_hw0 = [None] * len(file_list)
            self.orig_im_hw = [None] * len(file_list)
            self.orig_npy_files = [Path(f).with_suffix(".npy") for f in file_list]

        im, f, fn = self.orig_ims[i], file_list[i], self.orig_npy_files[i]

        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)
            else:  # read image
                im = imread(f, flags=self.cv2_flag)

            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw

            # 注意：这里不要先 resize 原图，否则后面的 crop_info 坐标就错了
            if im.ndim == 2:
                im = im[..., None]

            if self.augment:
                self.orig_ims[i], self.orig_im_hw0[i], self.orig_im_hw[i] = im, (h0, w0), im.shape[:2]

            return im, (h0, w0), im.shape[:2]

        return self.orig_ims[i], self.orig_im_hw0[i], self.orig_im_hw[i]

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writeable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        return self.get_image_and_label(index)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """返回一个原图 sample，其中包含多个 crop 子样本。 每个 crop 子样本都包含 img/cls/bboxes/batch_idx，可直接送入 criterion。.
        """
        label = deepcopy(self.labels[index])  # 原图级 label
        full_img, ori_shape, _ = self.load_image(index)
        h, w = ori_shape

        cls = label["cls"].astype(np.float32)  # (n,1)
        bboxes = label["bboxes"].astype(np.float32)  # (n,4), normalized xywh

        crop_size = self.crop_size
        stride = self.stride_crop
        min_box_wh = 1.0

        def build_crop_starts(full_len, crop_size, stride):
            if full_len <= crop_size:
                return [0]
            starts = list(range(0, full_len - crop_size + 1, stride))
            if starts[-1] != full_len - crop_size:
                starts.append(full_len - crop_size)
            return starts

        # 原图 normalized xywh -> absolute xyxy
        abs_boxes = np.zeros((len(bboxes), 4), dtype=np.float32)
        if len(bboxes):
            x_c = bboxes[:, 0] * w
            y_c = bboxes[:, 1] * h
            bw = bboxes[:, 2] * w
            bh = bboxes[:, 3] * h
            abs_boxes[:, 0] = x_c - bw / 2
            abs_boxes[:, 1] = y_c - bh / 2
            abs_boxes[:, 2] = x_c + bw / 2
            abs_boxes[:, 3] = y_c + bh / 2

        xs = build_crop_starts(w, crop_size, stride)
        ys = build_crop_starts(h, crop_size, stride)

        crop_samples = []

        for y1 in ys:
            for x1 in xs:
                x2 = min(x1 + crop_size, w)
                y2 = min(y1 + crop_size, h)
                cw = x2 - x1
                ch = y2 - y1

                if cw <= 0 or ch <= 0:
                    continue

                img = full_img[y1:y2, x1:x2]
                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    continue
                img = img.copy()

                crop_cls = []
                crop_boxes = []

                for i in range(len(abs_boxes)):
                    bx1, by1, bx2, by2 = abs_boxes[i]

                    ix1 = max(bx1, x1)
                    iy1 = max(by1, y1)
                    ix2 = min(bx2, x2)
                    iy2 = min(by2, y2)

                    iw = ix2 - ix1
                    ih = iy2 - iy1

                    if iw <= 0 or ih <= 0:
                        continue
                    if iw < min_box_wh or ih < min_box_wh:
                        continue

                    local_x1 = ix1 - x1
                    local_y1 = iy1 - y1
                    local_x2 = ix2 - x1
                    local_y2 = iy2 - y1

                    local_cx = (local_x1 + local_x2) / 2 / cw
                    local_cy = (local_y1 + local_y2) / 2 / ch
                    local_bw = (local_x2 - local_x1) / cw
                    local_bh = (local_y2 - local_y1) / ch

                    crop_cls.append(cls[i].copy())
                    crop_boxes.append([local_cx, local_cy, local_bw, local_bh])

                if len(crop_cls):
                    crop_cls = np.stack(crop_cls, axis=0).astype(np.float32)
                    crop_boxes = np.array(crop_boxes, dtype=np.float32)
                else:
                    crop_cls = np.zeros((0, 1), dtype=np.float32)
                    crop_boxes = np.zeros((0, 4), dtype=np.float32)

                crop_label = {
                    "im_file": label["im_file"],
                    "ori_shape": ori_shape,
                    "resized_shape": img.shape[:2],
                    "ratio_pad": (1.0, 1.0),
                    "img": img,
                    "cls": crop_cls,
                    "bboxes": crop_boxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "crop_info": (x1, y1, x2, y2),
                }

                # 复用你现有的 transform + update_labels_info
                crop_label = self.update_labels_info(crop_label)
                crop_label = self.transforms(crop_label)

                # 单张 crop 的 batch_idx 全设为 0
                if "batch_idx" in crop_label:
                    crop_label["batch_idx"] = torch.zeros_like(crop_label["batch_idx"])

                crop_samples.append(crop_label)

                # # =================== debug =====================
                # debug_img = img.copy()
                # for j in range(len(crop_boxes)):
                #     cx, cy, bw, bh = crop_boxes[j]
                #     x1b = int((cx - bw / 2) * cw)
                #     y1b = int((cy - bh / 2) * ch)
                #     x2b = int((cx + bw / 2) * cw)
                #     y2b = int((cy + bh / 2) * ch)
                #     cv2.rectangle(debug_img, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)

                # os.makedirs("/mnt/nfs_200T/optics/SHL/ultralytics/debug/debug_raw_crops", exist_ok=True)
                # cv2.imwrite(f"/mnt/nfs_200T/optics/SHL/ultralytics/debug/debug_raw_crops/index{index}_x{x1}_y{y1}.jpg", debug_img)
        return {
            "crop_samples": crop_samples,
            "num_crops": len(crop_samples),
            "im_file": label["im_file"],
        }

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError


class MidfusionBaseDataset(Dataset):
    """Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.

    Attributes:
        img_path (str): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float): Fraction of dataset to utilize.
        channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        cv2_flag (int): OpenCV flag for reading images.
        im_files (List[str]): List of image file paths.
        labels (List[Dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (List[Path]): List of numpy file paths.
        cache (str): Cache images to RAM or disk during training.
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Set the shape of bounding boxes as rectangles.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str | List[str]): Path to the folder containing images or list of image paths.
            imgsz (int): Image size for resizing.
            cache (bool | str): Cache images to RAM or disk during training.
            augment (bool): If True, data augmentation is applied.
            hyp (Dict[str, Any]): Hyperparameters to apply data augmentation.
            prefix (str): Prefix to print in log messages.
            rect (bool): If True, rectangular training is used.
            batch_size (int): Size of batches.
            stride (int): Stride used in the model.
            pad (float): Padding value.
            single_cls (bool): If True, single class training is used.
            classes (List[int], optional): List of included classes.
            fraction (float): Fraction of dataset to utilize.
            channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [[None] * self.ni] * len(self.im_files), [None] * self.ni, [None] * self.ni

        self.npy_files = [[] for _ in range(len(self.im_files))]
        for i in range(len(self.im_files)):
            self.npy_files[i] = [Path(f).with_suffix(".npy") for f in self.im_files[i]]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read image files from the specified path.

        Args:
            img_path (str | List[str]): Path or list of paths to image directories or files.

        Returns:
            (List[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f, im_files = [[] for _ in range(len(img_path))], [[] for _ in range(len(img_path))]  # image files
            for i, path in enumerate(img_path):
                for p in path if isinstance(path, list) else [path]:
                    p = Path(p)  # os-agnostic
                    if p.is_dir():  # dir
                        f[i] += glob.glob(str(p / "**" / "*.*"), recursive=True)
                        # F = list(p.rglob('*.*'))  # pathlib
                    elif p.is_file():  # file
                        with open(p, encoding="utf-8") as t:
                            t = t.read().strip().splitlines()
                            parent = str(p.parent) + os.sep
                            f[i] += [
                                x.replace("./", parent) if x.startswith("./") else x for x in t
                            ]  # local to global path
                            # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                    else:
                        raise FileNotFoundError(f"{self.prefix}{p} does not exist")
                im_files[i] = sorted(
                    x.replace("/", os.sep) for x in f[i] if x.rpartition(".")[-1].lower() in IMG_FORMATS
                )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """Update labels to include only specified classes.

        Args:
            include_class (List[int], optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (Tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (Tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        image = [[], []]
        # if self.ims[0][i] is None:  # not cached in RAM
        for m in range(len(self.im_files)):
            im, f, fn = self.ims[m][i], self.im_files[m][i], self.npy_files[m][i]
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    if any(
                        x in f for x in ("NDVI", "DoLP", "AoLP", "Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6", "Cam7")
                    ):
                        im = imread(f, flags=cv2.IMREAD_GRAYSCALE)
                    else:
                        im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                if any(
                    x in f for x in ("NDVI", "DoLP", "AoLP", "Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6", "Cam7")
                ):
                    im = imread(f, flags=cv2.IMREAD_GRAYSCALE)
                else:
                    im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[m][i], self.im_hw0[i], self.im_hw[i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[m][j], self.im_hw0[j], self.im_hw[j] = None, None, None
            if "Cam0" in f or "RGB" in f:
                image[0].append(im)
            else:
                image[1].append(im)
        for i in range(len(image)):
            image[i] = np.concatenate(image[i], axis=-1)
        image = np.concatenate(image, axis=-1)  # (H, W, C0 + C1)
        return image, (h0, w0), im.shape[:2]
        # else:
        #     for m in range(len(self.im_files)):
        #         image.append(self.ims[m][i])
        #     image = np.concatenate(image, axis=-1)
        #     return image, self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writeable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        for j in range(len(self.im_files)):
            self.im_files[j] = [self.im_files[j][i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (Dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError
