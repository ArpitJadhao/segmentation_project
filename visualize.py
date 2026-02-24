# visualize.py
"""
Visualize segmentation predictions with high-contrast colors.
Usage: python visualize.py --img path/to/rgb.png --mask path/to/mask_pred.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from config import CLASS_COLORS, IDX2NAME, NUM_CLASSES


def show_prediction(img_path: str, pred_mask_path: str):
    img   = np.array(Image.open(img_path).convert("RGB"))
    pred  = np.array(Image.open(pred_mask_path).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img);  axes[0].set_title("Original RGB");   axes[0].axis("off")
    axes[1].imshow(pred); axes[1].set_title("Prediction");     axes[1].axis("off")

    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i])/255, label=IDX2NAME[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",  required=True, help="Path to RGB image")
    parser.add_argument("--mask", required=True, help="Path to prediction PNG")
    args = parser.parse_args()
    show_prediction(args.img, args.mask)
