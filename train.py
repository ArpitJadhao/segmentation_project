# train.py
import os, random, numpy as np, torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from config import *
from dataset import DesertSegDataset
from model   import build_model
from utils   import IoUMetric, save_loss_iou_plots, compute_class_weights


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    set_seed(SEED)
    os.makedirs(RUNS_DIR, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using device: {device}")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA is not available. Falling back to CPU. Training will be significantly slower.")

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = DesertSegDataset(TRAIN_RGB, TRAIN_SEG, augment=True)
    val_ds   = DesertSegDataset(VAL_RGB,   VAL_SEG,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model().to(device)

    # ── Loss with class weights ──────────────────────────────────────────
    print("[INFO] Computing class weights from training masks...")
    class_weights = compute_class_weights(TRAIN_SEG).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = GradScaler('cuda', enabled=torch.cuda.is_available())

    best_miou    = 0.0
    patience_cnt = 0
    train_losses, val_losses, val_mious = [], [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(imgs)            # (B, C, H, W)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        iou_metric = IoUMetric(NUM_CLASSES)
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                    loss   = criterion(logits, masks)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                iou_metric.update(preds, masks)

        avg_val_loss = val_loss / len(val_loader)
        miou, per_class = iou_metric.compute()
        val_losses.append(avg_val_loss)
        val_mious .append(miou)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val mIoU: {miou:.4f}")

        # ── Per-class IoU log ─────────────────────────────────────────
        from config import IDX2NAME
        class_iou_str = " | ".join(
            f"{IDX2NAME[i]}: {per_class[i]:.3f}" if not np.isnan(per_class[i]) else f"{IDX2NAME[i]}: N/A"
            for i in range(NUM_CLASSES)
        )
        print(f"  Per-class IoU: {class_iou_str}")

        # ── Save best checkpoint ──────────────────────────────────────
        if miou > best_miou:
            best_miou    = miou
            patience_cnt = 0
            ckpt_path = os.path.join(RUNS_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": miou,
            }, ckpt_path)
            print(f"  ✓ New best mIoU: {best_miou:.4f} — checkpoint saved.")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # ── Save curves ───────────────────────────────────────────────────
    save_loss_iou_plots(train_losses, val_losses, val_mious, RUNS_DIR)
    print(f"\n[DONE] Training complete. Best Val mIoU: {best_miou:.4f}")
    print(f"       Best model at: {os.path.join(RUNS_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    train()
