# dataset.py
import os, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from config import IMAGE_SIZE
from utils import remap_mask


def get_image_mask_pairs(rgb_dir: str, seg_dir: str):
    """
    Match RGB images to segmentation masks by filename stem.
    Supports .png and .jpg for RGB; .png for masks.
    """
    rgb_files  = sorted(
        glob.glob(os.path.join(rgb_dir, "**", "*.png"), recursive=True) +
        glob.glob(os.path.join(rgb_dir, "**", "*.jpg"), recursive=True) +
        glob.glob(os.path.join(rgb_dir, "**", "*.PNG"), recursive=True)
    )
    seg_lookup = {}
    for p in glob.glob(os.path.join(seg_dir, "**", "*.png"), recursive=True):
        seg_lookup[os.path.splitext(os.path.basename(p))[0]] = p

    pairs = []
    for rgb_path in rgb_files:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        if stem in seg_lookup:
            pairs.append((rgb_path, seg_lookup[stem]))
        else:
            print(f"[WARN] No matching mask for: {rgb_path}")
    return pairs


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DesertSegDataset(Dataset):
    def __init__(self, rgb_dir: str, seg_dir: str, augment: bool = False):
        self.pairs   = get_image_mask_pairs(rgb_dir, seg_dir)
        self.augment = augment
        self.size    = IMAGE_SIZE
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image-mask pairs found!\n  RGB dir: {rgb_dir}\n  SEG dir: {seg_dir}\n"
                "Check dataset folder structure matches plan.md."
            )
        print(f"[INFO] Dataset loaded: {len(self.pairs)} pairs from {rgb_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, seg_path = self.pairs[idx]

        img  = Image.open(rgb_path).convert("RGB")
        raw  = np.array(Image.open(seg_path))          # raw class IDs
        mask = remap_mask(raw)                          # → 0–9 indices
        mask = Image.fromarray(mask.astype(np.int32))  # PIL for transforms

        # ── Resize ──────────────────────────────────────────────────────
        img  = img.resize ((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        # ── Augmentation ────────────────────────────────────────────────
        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.8:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            # Random resized crop
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.5, 1.0), ratio=(0.75, 1.33)
            )
            img  = TF.resized_crop(img,  i, j, h, w, (self.size, self.size), Image.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, (self.size, self.size), Image.NEAREST)
            # Color jitter (image only)
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            img = jitter(img)

        # ── To tensor ───────────────────────────────────────────────────
        img_t  = T.ToTensor()(img)
        img_t  = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_t)
        mask_t = torch.from_numpy(np.array(mask)).long()

        return img_t, mask_t


class TestDataset(Dataset):
    """RGB-only dataset for testImages (no ground truth)."""
    def __init__(self, rgb_dir: str):
        self.paths = sorted(
            glob.glob(os.path.join(rgb_dir, "**", "*.png"), recursive=True) +
            glob.glob(os.path.join(rgb_dir, "**", "*.jpg"), recursive=True)
        )
        self.size  = IMAGE_SIZE
        print(f"[INFO] Test dataset: {len(self.paths)} images from {rgb_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR
        )
        img_t = T.ToTensor()(img)
        img_t = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])(img_t)
        return img_t, path
