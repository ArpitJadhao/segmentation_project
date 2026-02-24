# utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from config import ID2IDX, NUM_CLASSES, DEFAULT_IDX, CLASS_COLORS, IDX2NAME


def remap_mask(raw_mask_np: np.ndarray) -> np.ndarray:
    """
    Convert a mask with raw class IDs (100, 200, ...) to sequential indices (0–9).
    Unknown values → DEFAULT_IDX (8 = Landscape).
    raw_mask_np: 2D numpy array of dtype uint16 or int32
    """
    out = np.full(raw_mask_np.shape, DEFAULT_IDX, dtype=np.int64)
    for raw_id, idx in ID2IDX.items():
        out[raw_mask_np == raw_id] = idx
    return out


class IoUMetric:
    """
    Accumulates TP, FP, FN across batches for per-class and mean IoU.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes, dtype=np.float64)
        self.union        = np.zeros(self.num_classes, dtype=np.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred:   (B, H, W) long tensor — predicted class indices
        target: (B, H, W) long tensor — ground truth class indices
        """
        pred   = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        for cls in range(self.num_classes):
            pred_cls   = pred   == cls
            target_cls = target == cls
            self.intersection[cls] += np.logical_and(pred_cls, target_cls).sum()
            self.union[cls]        += np.logical_or (pred_cls, target_cls).sum()

    def compute(self):
        iou_per_class = np.where(
            self.union > 0,
            self.intersection / self.union,
            np.nan
        )
        miou = np.nanmean(iou_per_class)
        return miou, iou_per_class


def colorize_mask(mask_idx: np.ndarray) -> np.ndarray:
    """
    mask_idx: (H, W) numpy array of class indices 0–9
    Returns:  (H, W, 3) uint8 RGB image
    """
    rgb = np.zeros((*mask_idx.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        rgb[mask_idx == idx] = color
    return rgb


def save_loss_iou_plots(train_losses, val_losses, val_mious, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss", color="blue")
    axes[0].plot(epochs, val_losses,   label="Val Loss",   color="orange")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, val_mious, label="Val mIoU", color="green")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to {save_dir}/training_curves.png")


def compute_class_weights(seg_dir: str, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from all masks in seg_dir.
    Returns a float32 tensor of shape (num_classes,).
    """
    from PIL import Image
    import glob
    counts = np.zeros(num_classes, dtype=np.float64)
    mask_paths = glob.glob(os.path.join(seg_dir, "**", "*.png"), recursive=True)
    mask_paths += glob.glob(os.path.join(seg_dir, "**", "*.PNG"), recursive=True)
    if not mask_paths:
        print("[WARN] No masks found for class weight computation. Using uniform weights.")
        return torch.ones(num_classes, dtype=torch.float32)
    for p in mask_paths:
        raw = np.array(Image.open(p))
        remapped = remap_mask(raw)
        for cls in range(num_classes):
            counts[cls] += (remapped == cls).sum()
    total = counts.sum()
    freq  = counts / (total + 1e-8)
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalize
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, inputs, targets):
        ce = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()
