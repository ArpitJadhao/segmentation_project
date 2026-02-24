# model.py
import torch
from config import NUM_CLASSES, ENCODER, ENCODER_WEIGHTS, MODEL_NAME


def build_model() -> torch.nn.Module:
    """
    Build DeepLabV3+ with MobileNetV3-Large backbone.
    Falls back gracefully with a clear error if smp is not installed.
    """
    try:
        import segmentation_models_pytorch as smp
        model = smp.DeepLabV3Plus(
            encoder_name    = ENCODER,
            encoder_weights = ENCODER_WEIGHTS,
            in_channels     = 3,
            classes         = NUM_CLASSES,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to build model: {e}\n"
            "Ensure segmentation_models_pytorch is installed:\n"
            "  pip install segmentation-models-pytorch"
        )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str,
                    device: torch.device) -> torch.nn.Module:
    # weights_only=False is used because checkpoints may contain numpy types 
    # that are now restricted by default in newer PyTorch versions.
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    return model
