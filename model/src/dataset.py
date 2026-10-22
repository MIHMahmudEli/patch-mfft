"""
Patch-MFFT Dataset Pipeline with Online Splicing and Generative Inpainting Augmentation.

Generates realistic forensic composite attacks during training:
1. Two-Source Splicing (Blackmail / Extortion Simulation):
   Extracts irregular patches/contours from a secondary source image (Camera B)
   and splices them into the primary host image (Camera A) with edge feathering.
2. Generative Inpainting Simulation:
   Masks out an irregular region and fills it with generative-style diffusion
   upsampling artifacts.
3. Authentic Photographic Baseline (Label 0):
   Untouched camera sensor capture with empty ground-truth mask.
4. Fully AI-Generated (Label 1):
   End-to-end synthesis (Diffusion, GANs, AI Cartoons/Anime) with full mask.
5. Locally Spliced / Inpainted (Label 2):
   Partial manipulation with binary ground-truth tampering mask.
"""

import math
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OnlineTamperingAugmentor:
    """
    On-the-fly composite splicing and generative inpainting generator.
    Creates realistic tampering boundaries, phase discontinuities, and sensor noise mismatches.
    """

    def __init__(self, img_size: int = 384, p_splice: float = 0.5, p_inpaint: float = 0.5):
        self.img_size = img_size
        self.p_splice = p_splice
        self.p_inpaint = p_inpaint

    def generate_irregular_mask(self, width: int, height: int) -> Image.Image:
        """Generates random polygon, ellipse, or brush-stroke masks."""
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        mask_type = random.choice(["polygon", "ellipse", "multi_patch"])

        if mask_type == "polygon":
            n_points = random.randint(4, 8)
            cx, cy = random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4)
            r_max = random.randint(width // 6, width // 3)
            points = []
            for i in range(n_points):
                angle = 2 * math.pi * i / n_points + random.uniform(-0.3, 0.3)
                r = r_max * random.uniform(0.6, 1.0)
                px = max(0, min(width - 1, int(cx + r * math.cos(angle))))
                py = max(0, min(height - 1, int(cy + r * math.sin(angle))))
                points.append((px, py))
            draw.polygon(points, fill=255)

        elif mask_type == "ellipse":
            cx, cy = random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4)
            rx, ry = random.randint(width // 8, width // 3), random.randint(height // 8, height // 3)
            box = [max(0, cx - rx), max(0, cy - ry), min(width, cx + rx), min(height, cy + ry)]
            draw.ellipse(box, fill=255)

        else:
            # Multi-patch stroke
            for _ in range(random.randint(2, 4)):
                x1, y1 = random.randint(0, width), random.randint(0, height)
                x2, y2 = x1 + random.randint(-60, 60), y1 + random.randint(-60, 60)
                draw.line([(x1, y1), (x2, y2)], fill=255, width=random.randint(20, 50))

        # Soft edge feathering simulating realistic blending
        if random.random() > 0.3:
            blur_radius = random.uniform(1.0, 3.0)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return mask

    def apply_splicing(self, host_img: Image.Image, donor_img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """Splices an irregular region from donor_img into host_img."""
        w, h = host_img.size
        donor_resized = donor_img.resize((w, h), Image.Resampling.BILINEAR)

        mask_pil = self.generate_irregular_mask(w, h)
        spliced_img = Image.composite(donor_resized, host_img, mask_pil)

        mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0
        binary_mask = (mask_arr > 0.1).astype(np.float32)
        return spliced_img, binary_mask

    def apply_inpainting_simulation(self, img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """Simulates localized diffusion inpainting by altering patch high-frequency spectrum."""
        w, h = img.size
        mask_pil = self.generate_irregular_mask(w, h)
        mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0

        img_np = np.array(img, dtype=np.float32)

        # Inpainted content simulation: smooth gradient + high-frequency upsampling grid artifact
        smoothed = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 5.0)))
        smoothed_np = np.array(smoothed, dtype=np.float32)

        # Inject synthetic checkerboard pattern simulating transposed convolution
        yy, xx = np.indices((h, w))
        checkerboard = (((xx // 2) + (yy // 2)) % 2).astype(np.float32) * random.uniform(8.0, 20.0)
        checkerboard = np.stack([checkerboard] * 3, axis=-1)

        simulated_patch = np.clip(smoothed_np + checkerboard, 0, 255)

        # Blend
        alpha = np.expand_dims(mask_arr, axis=-1)
        blended = (img_np * (1.0 - alpha) + simulated_patch * alpha).astype(np.uint8)

        binary_mask = (mask_arr > 0.1).astype(np.float32)
        return Image.fromarray(blended), binary_mask


class TamperingForensicsDataset(Dataset):
    """
    Forensic Dataset supporting 3 classes and pixel-level localization:
    - 0: Authentic Camera Photography
    - 1: Fully AI-Generated (Text-to-Image / AI Cartoon)
    - 2: Locally Spliced / Inpainted Tampering
    """

    def __init__(
        self,
        samples: List[dict],
        img_size: int = 384,
        is_train: bool = True,
        online_tamper_prob: float = 0.35,
    ):
        self.samples = samples
        self.img_size = img_size
        self.is_train = is_train
        self.online_tamper_prob = online_tamper_prob
        self.augmentor = OnlineTamperingAugmentor(img_size=img_size)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Union[str, Path]) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            # Fallback black canvas if file read error occurs
            return Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

    def _preprocess(self, image: Image.Image, mask: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Aspect-ratio preserving resize + center crop
        w, h = image.size
        target = self.img_size + 16
        if w < h:
            new_w, new_h = target, max(target, int(h * target / w))
        else:
            new_h, new_w = target, max(target, int(w * target / h))

        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        left = max(0, (new_w - self.img_size) // 2)
        top = max(0, (new_h - self.img_size) // 2)
        image = image.crop((left, top, left + self.img_size, top + self.img_size))

        img_tensor = transforms.functional.to_tensor(image)
        img_tensor = self.normalize(img_tensor)

        if mask is not None:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.NEAREST)
            mask_pil = mask_pil.crop((left, top, left + self.img_size, top + self.img_size))
            mask_tensor = torch.from_numpy(np.array(mask_pil, dtype=np.float32) / 255.0).unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        return img_tensor, mask_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        item = self.samples[idx]
        img_path = item["path"]
        label = int(item["label"])  # 0: real, 1: full-ai, 2: tampered
        image = self._load_image(img_path)
        mask = None

        # If sample already has an existing ground truth mask path
        if "mask_path" in item and item["mask_path"] and Path(item["mask_path"]).exists():
            try:
                mask_img = Image.open(item["mask_path"]).convert("L")
                mask = np.array(mask_img, dtype=np.float32) / 255.0
            except Exception:
                mask = None

        # During training, dynamically synthesize realistic splicing/inpainting
        if self.is_train and random.random() < self.online_tamper_prob:
            # Pick a donor sample
            donor_idx = random.randint(0, len(self.samples) - 1)
            donor_img = self._load_image(self.samples[donor_idx]["path"])

            if random.random() > 0.5:
                image, mask = self.augmentor.apply_splicing(image, donor_img)
            else:
                image, mask = self.augmentor.apply_inpainting_simulation(image)

            label = 2  # Tampered class

        # Set default masks if not tampered
        if mask is None:
            if label == 1:
                mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
            else:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        img_tensor, mask_tensor = self._preprocess(image, mask)
        return img_tensor, label, mask_tensor
