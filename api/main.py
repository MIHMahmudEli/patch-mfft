"""
Patch-MFFT Tampering Forensics API
==================================
FastAPI service providing:
- 3-Class Forensics: Authentic vs Full-AI vs Spliced/Inpainted Tampering
- Real-time Pixel-Level Tampering Localization Masks (Base64 PNG)
- Radial Fourier Spectral Band Energy Decomposition
"""

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import HealthResponse, ModelsResponse, PredictionResponse, ModelInfo
from .model_server import TamperingModelServer

MODEL_INFO = {
    "tiny":  {"params": "563K",  "description": "Fastest - edge/mobile tampering detection profile"},
    "base":  {"params": "3.28M", "description": "Balanced accuracy, splicing detection, and localization"},
    "large": {"params": "14.78M","description": "Highest forensic precision for inpainting & composite extortion"},
}

model_registry: Dict[str, TamperingModelServer] = {}
DEFAULT_VARIANT = os.environ.get("PATCH_MFFT_VARIANT", "base")
IMG_SIZE = int(os.environ.get("PATCH_MFFT_IMG_SIZE", "384"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Initializing Patch-MFFT model registry...")
    ckpt_dir_env = os.environ.get("PATCH_MFFT_CHECKPOINT_DIR")
    if ckpt_dir_env and Path(ckpt_dir_env).exists():
        ckpt_dir = Path(ckpt_dir_env)
    elif (Path(__file__).parent.parent / "checkpoints").exists():
        ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    else:
        ckpt_dir = Path(__file__).parent.parent / "model" / "checkpoints"

    for variant in ["tiny", "base", "large"]:
        ckpt_path = ckpt_dir / f"{variant}.pt"
        try:
            model_registry[variant] = TamperingModelServer(
                str(ckpt_path) if ckpt_path.exists() else None,
                variant=variant,
                img_size=IMG_SIZE,
            )
        except Exception as e:
            print(f"[startup] Failed to initialize variant '{variant}': {e}")
    print(f"[startup] Loaded variants: {list(model_registry.keys())}")
    yield
    model_registry.clear()


app = FastAPI(
    title="Patch-MFFT Tampering Forensics API",
    description="Multi-Frequency Transformer for Localized Inpainting, Two-Person Splicing, and Composite Blackmail Detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=list(model_registry.keys()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(
        default=DEFAULT_VARIANT if DEFAULT_VARIANT in model_registry else list(model_registry.keys())[0],
        models=[
            ModelInfo(
                id=v,
                loaded=v in model_registry,
                params=MODEL_INFO[v]["params"],
                description=MODEL_INFO[v]["description"],
            )
            for v in MODEL_INFO
        ],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = "large",
):
    variant = (model or DEFAULT_VARIANT).lower()
    if variant not in model_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model variant '{variant}' not found. Available: {list(model_registry.keys())}",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size exceeds 25 MB limit.")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted image file.")

    server = model_registry[variant]
    result = server.predict(image)

    return PredictionResponse(**result)
