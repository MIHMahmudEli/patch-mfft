"""
Patch-MFFT Standalone CLI Inference & Tampering Localization Visualizer.
Detects generative inpainting, composite forgery, and style tampering in digital images.

Usage:
    python predict.py --image path/to/image.jpg --model base --output output_heatmap.png
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.cm as cm

# Add model to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "model" / "src"))

from model import build_patch_mfft


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-MFFT Image Tampering Detection & Localization CLI")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image file (JPEG, PNG, WEBP)")
    parser.add_argument("--model", "-m", type=str, default="base", choices=["tiny", "base", "large"], help="Patch-MFFT model variant")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Custom path to model weights .pt file (optional)")
    parser.add_argument("--output", "-o", type=str, default="output_heatmap.png", help="Path to save output heatmap overlay")
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(variant: str, weights_path: Path, device: torch.device):
    model = build_patch_mfft(variant=variant, img_size=384, num_classes=3)
    if weights_path.exists():
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Strip potential _orig_mod prefix from compile
        cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned_state, strict=True)
        print(f"[Model] Successfully loaded weights with strict=True from: {weights_path}")
    else:
        print(f"[Warning] Weights file not found at {weights_path}. Please run:")
        print(f"          python scripts/download_checkpoints.py {variant}")
        sys.exit(1)

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = get_device(args.device)
    print("=== Patch-MFFT Forensic Inference Engine ===")
    print(f"Device:           {device}")
    print(f"Model Variant:    patch_mfft_{args.model}")

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    # Locate weights
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = BASE_DIR / "model" / "checkpoints" / f"{args.model}.pt"

    model = load_model(args.model, weights_path, device)

    # Load and preprocess image matching training pipeline
    orig_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    transform = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(orig_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        cls_logits, seg_logits = model(tensor, return_mask=True)
        probs = F.softmax(cls_logits, dim=-1).squeeze(0)
        pred_idx = torch.argmax(probs).item()

        # Dense tampering mask
        raw_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()

    classes = [
        ("AUTHENTIC CAMERA CAPTURE", "No manipulation detected. Pristine optical sensor noise."),
        ("HOLISTIC GENERATIVE AI", "End-to-end synthetic image synthesized by diffusion / GAN generator."),
        ("SPLICED / INPAINTED TAMPERING", "Localized digital tampering detected. Boundary phase incoherence.")
    ]

    verdict_label, verdict_desc = classes[pred_idx]
    confidence = probs[pred_idx].item() * 100.0

    print("\n--- Forensic Analysis Results ---")
    print(f"Decision:         {verdict_label}")
    print(f"Confidence:       {confidence:.2f}%")
    print(f"Authentic Prob:   {probs[0].item() * 100:.2f}%")
    print(f"Full AI Prob:     {probs[1].item() * 100:.2f}%")
    print(f"Spliced Prob:     {probs[2].item() * 100:.2f}%")

    # Forensic mask gating
    if pred_idx == 0:
        mask_prob = np.zeros_like(raw_mask)
    elif pred_idx == 1:
        mask_prob = np.clip(raw_mask, 0.0, 1.0)
    else:
        mask_prob = np.where(raw_mask > 0.20, raw_mask, 0.0)

    print(f"Peak Mask Value:  {mask_prob.max() * 100:.2f}%")

    # Generate Heatmap Overlay
    heatmap_colored = cm.jet(mask_prob)[:, :, :3]  # Drop alpha
    heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)

    # Blend original and heatmap
    overlay = Image.blend(orig_img, heatmap_img, alpha=0.45 if pred_idx != 0 else 0.0)
    out_path = Path(args.output)
    overlay.save(out_path)
    print(f"\nSaved localization heatmap overlay to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
