"""
Model Server for Patch-MFFT Tampering Forensics.
Handles inference, aspect-ratio preserving preprocessing,
3-class prediction, and pixel-level tampering localization masks.
"""

import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "model" / "src"))
from model import PatchMFFT, build_patch_mfft


class TamperingModelServer:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        variant: str = "large",
        img_size: int = 384,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.variant = variant
        self.img_size = img_size
        print(f"[TamperingModelServer] Initializing Patch-MFFT ({variant}) on {self.device}")

        self.model = build_patch_mfft(variant, img_size=img_size, num_classes=3)
        self.model = self.model.to(self.device)
        self.is_loaded = False

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        else:
            print(f"[TamperingModelServer] Running with initialized weights (no checkpoint at {checkpoint_path})")

        self.model.eval()
        self.warmup()

    def load_checkpoint(self, path: str):
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            state = torch.load(path, map_location=self.device, weights_only=False)

        raw_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        self.model.load_state_dict(raw_state, strict=False)
        self.is_loaded = True
        print(f"[TamperingModelServer] Successfully loaded weights from {path}")

    def warmup(self, iters: int = 2):
        dummy = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        with torch.no_grad():
            for _ in range(iters):
                _ = self.model(dummy, return_mask=True)
        print(f"[TamperingModelServer] Warmup completed ({iters} iterations).")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        w, h = image.size
        target = self.img_size + 16
        if w < h:
            new_w, new_h = target, max(target, int(h * target / w))
        else:
            new_h, new_w = target, max(target, int(w * target / h))
        image = image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
        left = max(0, (new_w - self.img_size) // 2)
        top = max(0, (new_h - self.img_size) // 2)
        image = image.crop((left, top, left + self.img_size, top + self.img_size))

        img_array = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = ((img_array - mean) / std).astype(np.float32)

        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start = time.time()
        tensor = self._preprocess(image)

        logits, mask = self.model(tensor, return_mask=True)
        probs = F.softmax(logits, dim=-1)[0]

        pred_idx = torch.argmax(probs).item()
        classes = ["authentic", "full_ai", "spliced_tampered"]
        prediction_label = classes[pred_idx]

        class_probs = {
            "authentic": round(probs[0].item(), 4),
            "full_ai": round(probs[1].item(), 4),
            "spliced_tampered": round(probs[2].item(), 4),
        }

        # Mask to Base64 PNG (apply sigmoid to convert raw logits to probabilities)
        mask_np = torch.sigmoid(mask)[0, 0].cpu().numpy()
        mask_uint8 = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode="L")
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Frequency bands
        bands = self.model.decomposer(tensor)
        band_mags = [float(b.abs().mean().item()) for b in bands]
        band_dict = {
            f"band_{i+1}": round(band_mags[i], 4) for i in range(len(band_mags))
        }

        elapsed_ms = (time.time() - start) * 1000.0

        return {
            "prediction": prediction_label,
            "confidence": round(probs[pred_idx].item(), 4),
            "tampering_detected": pred_idx in [1, 2],
            "class_probabilities": class_probs,
            "tampering_heatmap_b64": mask_b64,
            "frequency_band_contributions": band_dict,
            "processing_time_ms": round(elapsed_ms, 2),
            "model_variant": self.variant,
        }
