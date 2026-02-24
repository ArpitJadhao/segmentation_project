# test.py
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.amp import autocast

from config  import *
from dataset import TestDataset, DesertSegDataset
from model   import build_model, load_checkpoint
from utils   import IoUMetric, colorize_mask


def run_test():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(RUNS_DIR, "best_model.pth")
    out_dir   = os.path.join(RUNS_DIR, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Run train.py first."
        )

    # ── Load model ───────────────────────────────────────────────────
    model = build_model()
    model = load_checkpoint(model, ckpt_path, device)
    model = model.to(device).eval()

    # ── Inference on testImages/ ─────────────────────────────────────
    test_ds     = TestDataset(TEST_RGB)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"\n[INFO] Running inference on {len(test_ds)} test images...")
    with torch.no_grad():
        for imgs, paths in test_loader:
            imgs = imgs.to(device)
            with autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()[0]  # (H, W)

            colored = colorize_mask(preds)
            out_name = os.path.splitext(os.path.basename(paths[0]))[0] + "_pred.png"
            Image.fromarray(colored).save(os.path.join(out_dir, out_name))

    print(f"[INFO] Predictions saved to: {out_dir}")

    # ── Optional: compute IoU on Val set (ground truth available) ───
    print("\n[INFO] Computing IoU on Val set for benchmark...")
    val_ds     = DesertSegDataset(VAL_RGB, VAL_SEG, augment=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)
    iou_metric = IoUMetric(NUM_CLASSES)

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(imgs)
            preds = logits.argmax(dim=1)
            iou_metric.update(preds, masks)

    miou, per_class = iou_metric.compute()
    print(f"\n{'='*50}")
    print(f"  Val mIoU: {miou:.4f}")
    print(f"{'='*50}")
    for i in range(NUM_CLASSES):
        val = per_class[i]
        name = IDX2NAME[i]
        print(f"  {name:<20}: {val:.4f}" if not np.isnan(val) else f"  {name:<20}: N/A (no samples)")
    print(f"{'='*50}\n")

    # Save IoU report
    report_path = os.path.join(RUNS_DIR, "iou_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Val mIoU: {miou:.4f}\n\nPer-class IoU:\n")
        for i in range(NUM_CLASSES):
            val = per_class[i]
            f.write(f"  {IDX2NAME[i]:<20}: {val:.4f if not np.isnan(val) else 'N/A'}\n")
    print(f"[INFO] IoU report saved to {report_path}")


if __name__ == "__main__":
    run_test()
