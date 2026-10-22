"""
Patch-MFFT: Dual-Branch Radial Frequency Transformer for
Localized Generative Inpainting, Splicing Extortion, and Style Tampering Forensics.

Core Architecture (as validated in production swarm training):
1. Radial 2D-FFT Annular Sub-band Decomposition (Low, Mid, High concentric bands).
2. Multi-Band Depthwise-Separable Feature Encoders (encoders.0, encoders.1, encoders.2).
3. Cross-Attention Frequency Fusion (mha + LayerNorm blocks).
4. Dual-Branch Spatial-Max Gating (SMG): Unifies Global Average Pooling and Spatial Max-Pooling.
5. Dual Task Decoupled Output:
   - 3-Class Triage Categorical Head:
     * 0: Authentic Camera Photography
     * 1: Holistic Generative AI Synthesis
     * 2: Locally Spliced / Inpainted / Tampered
   - Pixel-Level Tampering Localization Decoder:
     * High-resolution tampering anomaly heatmap [0, 1]^{H x W}.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialFourierDecomposition(nn.Module):
    """
    Decomposes an input image into K concentric radial annular frequency bands
    in the 2D Fast Fourier Transform (FFT) domain.
    """

    def __init__(self, img_size: int = 384, num_bands: int = 3):
        super().__init__()
        self.img_size = img_size
        self.num_bands = num_bands
        configs = (
            [(0.0, 0.15), (0.15, 0.5), (0.5, 1.0)]
            if num_bands == 3
            else [(0.0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.0)]
        )
        y = torch.linspace(-1, 1, img_size)
        x = torch.linspace(-1, 1, img_size)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(gx**2 + gy**2) / math.sqrt(2)
        masks = []
        for low_r, high_r in configs:
            m = (r >= low_r) & (r <= high_r if high_r >= 1.0 else r < high_r)
            masks.append(m.float())
        stacked = torch.stack(masks, dim=0)
        norm = stacked.sum(dim=0, keepdim=True).clamp(min=1e-8)
        self.register_buffer("masks", stacked / norm)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        X_fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        bands = []
        for k in range(self.num_bands):
            m = self.masks[k].unsqueeze(0).unsqueeze(0)
            bands.append(
                torch.fft.ifft2(
                    torch.fft.ifftshift(X_fft * m, dim=(-2, -1)), norm="ortho"
                ).real
            )
        return bands


class DCT2DDecomposition(nn.Module):
    def __init__(self, img_size: int = 384, num_bands: int = 3):
        super().__init__()
        self.img_size = img_size
        self.num_bands = num_bands
        N = img_size
        n = torch.arange(N).float()
        k = torch.arange(N).float().unsqueeze(1)
        D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N)) * math.sqrt(2.0 / N)
        D[0, :] = D[0, :] / math.sqrt(2.0)
        self.register_buffer("D", D)

        u = torch.linspace(0, 1, N)
        v = torch.linspace(0, 1, N)
        gu, gv = torch.meshgrid(u, v, indexing="ij")
        d = (gu + gv) / 2.0
        configs = [(0.0, 0.15), (0.15, 0.5), (0.5, 1.0)]
        masks = []
        for low_d, high_d in configs:
            m = (d >= low_d) & (d <= high_d if high_d >= 1.0 else d < high_d)
            masks.append(m.float())
        stacked = torch.stack(masks, dim=0)
        norm = stacked.sum(dim=0, keepdim=True).clamp(min=1e-8)
        self.register_buffer("masks", stacked / norm)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        D = self.D
        xD_T = torch.matmul(x, D.t())
        Y = torch.matmul(D, xD_T)
        bands = []
        for k in range(self.num_bands):
            m = self.masks[k].unsqueeze(0).unsqueeze(0)
            Y_k = Y * m
            D_T_Y = torch.matmul(D.t(), Y_k)
            bands.append(torch.matmul(D_T_Y, D))
        return bands


class HaarDWTDecomposition(nn.Module):
    def __init__(self, in_c: int = 3):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) * 0.5
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)
        self.in_c = in_c

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        bands = []
        for i in range(4):
            filt = self.filters[i : i + 1].repeat(C, 1, 1, 1)
            y = F.conv2d(x, filt, stride=2, groups=C)
            y_up = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            bands.append(y_up)
        # 3 bands: Low (LL), Mid (LH + HL), High (HH)
        return [bands[0], (bands[1] + bands[2]) * 0.5, bands[3]]


class SpatialOnlyDecomposition(nn.Module):
    def __init__(self, num_bands: int = 3):
        super().__init__()
        self.num_bands = num_bands

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [x] * self.num_bands


class DepthwiseSeparableConvBlock(nn.Module):
    """Parameter-efficient depthwise separable convolutional residual block."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.dw2 = nn.Conv2d(out_c, out_c, 3, 1, 1, groups=out_c, bias=False)
        self.pw2 = nn.Conv2d(out_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.res = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
            if (stride != 1 or in_c != out_c)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.pw(self.dw(x))))
        out = self.bn2(self.pw2(self.dw2(out)))
        return self.act(out + self.res(x))


class UniversalPoolingModule(nn.Module):
    """
    Universal feature aggregator supporting Dual-Branch Spatial-Max Gating (SMG),
    Global Average Pooling (GAP), Global Max-Pooling (GMP), and GeM.
    """

    def __init__(self, in_f: int, out_f: int, mode: str = "smg"):
        super().__init__()
        self.mode = mode
        if mode == "smg":
            self.proj = nn.Sequential(
                nn.Linear(in_f * 2, out_f),
                nn.BatchNorm1d(out_f),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.GELU(),
                nn.Dropout(0.1),
            )

    def forward(self, x: torch.Tensor, cls_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "smg":
            avg_p = x.mean(dim=[-2, -1])
            max_p = x.amax(dim=[-2, -1])
            return self.proj(torch.cat([avg_p, max_p], dim=-1))
        elif self.mode == "avg":
            return self.proj(x.mean(dim=[-2, -1]))
        elif self.mode == "max":
            return self.proj(x.amax(dim=[-2, -1]))
        elif self.mode == "gem":
            p = 3.0
            gem_p = (x.clamp(min=1e-6) ** p).mean(dim=[-2, -1]) ** (1.0 / p)
            return self.proj(gem_p)
        elif self.mode == "cls":
            return self.proj(cls_token)
        else:
            return self.proj(x.mean(dim=[-2, -1]))


class TamperingLocalizationDecoder(nn.Module):
    """
    Lightweight convolutional decoder for dense pixel-level tampering segmentation.
    Produces uncompressed 1 x H x W probability maps.
    """

    def __init__(self, in_c: int, img_size: int = 384):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Upsample(size=(img_size, img_size), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(x)


class PatchMFFT(nn.Module):
    """
    Patch-MFFT: Multi-Frequency Fusion Transformer for Image Tampering Forensics.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        depth: int = 3,
        num_bands: int = 3,
        img_size: int = 384,
        num_classes: int = 3,
        ablation: Optional[dict] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        self.img_size = img_size
        self.ablation = ablation or {}

        t_type = self.ablation.get("transform", "radial_fft")
        if self.ablation.get("spatial_only", False) or t_type == "spatial":
            self.decomposer = SpatialOnlyDecomposition(num_bands)
        elif t_type == "dct":
            self.decomposer = DCT2DDecomposition(img_size, num_bands)
        elif t_type == "dwt":
            self.decomposer = HaarDWTDecomposition(3)
        else:
            self.decomposer = RadialFourierDecomposition(img_size, num_bands)

        # 1. Sub-band convolutional encoders (stride 2 x 4 stages = downscale 16x)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConvBlock(3, 32, 2),
                DepthwiseSeparableConvBlock(32, 64, 2),
                DepthwiseSeparableConvBlock(64, 128, 2),
                DepthwiseSeparableConvBlock(128, embed_dim, 2),
            )
            for _ in range(num_bands)
        ])

        # 2. Multi-head cross-attention & LayerNorm across bands
        self.mha = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])

        # 3. Dual-Branch SMG pooling module
        pool_mode = self.ablation.get("pool_mode", "smg")
        if self.ablation.get("no_smg", False):
            pool_mode = "avg"
        self.pool_mode = pool_mode
        self.smg = UniversalPoolingModule(embed_dim, embed_dim, mode=pool_mode)

        if pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # 4. 3-Class Categorical Head (Authentic, Full AI, Spliced/Tampered)
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # 5. Tampering Localization Decoder
        self.loc = TamperingLocalizationDecoder(embed_dim, img_size)

    @property
    def pool(self) -> UniversalPoolingModule:
        return self.smg

    def forward(
        self,
        x: torch.Tensor,
        return_mask: bool = False,
        return_heatmap: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = x.shape[0]
        bands = self.decomposer(x)
        feats = [enc(b) for enc, b in zip(self.encoders, bands)]
        tokens = torch.cat([f.flatten(2).permute(0, 2, 1) for f in feats], dim=1)

        if self.cls_token is not None:
            cls_t = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_t, tokens], dim=1)
            for mha, norm in zip(self.mha, self.norms):
                attn, _ = mha(tokens, tokens, tokens, need_weights=False)
                tokens = norm(tokens + attn)
            extracted_cls = tokens[:, 0, :]
            n_p = (self.img_size // 16) ** 2
            fused = (
                tokens[:, -n_p:, :]
                .permute(0, 2, 1)
                .reshape(B, self.embed_dim, self.img_size // 16, self.img_size // 16)
            )
            pool = self.smg(fused, cls_token=extracted_cls)
        else:
            for mha, norm in zip(self.mha, self.norms):
                attn, _ = mha(tokens, tokens, tokens, need_weights=False)
                tokens = norm(tokens + attn)
            n_p = (self.img_size // 16) ** 2
            fused = (
                tokens[:, -n_p:, :]
                .permute(0, 2, 1)
                .reshape(B, self.embed_dim, self.img_size // 16, self.img_size // 16)
            )
            pool = self.smg(fused)

        logits = self.cls(pool)

        if return_mask or return_heatmap:
            mask = self.loc(fused)
            return logits, mask

        return logits


def build_patch_mfft(
    variant: str = "base",
    img_size: int = 384,
    num_classes: int = 3,
    ablation: Optional[dict] = None,
) -> PatchMFFT:
    """
    Factory builder for Patch-MFFT model variants:
    - tiny:  embed_dim=128, heads=4,  depth=2 (563K params)
    - base:  embed_dim=384, heads=6,  depth=3 (3.28M params)
    - large: embed_dim=768, heads=12, depth=4 (13.94M params)
    """
    cfgs = {
        "tiny": {"embed_dim": 128, "heads": 4, "depth": 2},
        "base": {"embed_dim": 384, "heads": 6, "depth": 3},
        "large": {"embed_dim": 768, "heads": 12, "depth": 4},
    }
    c = cfgs.get(variant, cfgs["base"])
    return PatchMFFT(
        embed_dim=c["embed_dim"],
        num_heads=c["heads"],
        depth=c["depth"],
        img_size=img_size,
        num_classes=num_classes,
        ablation=ablation,
    )
