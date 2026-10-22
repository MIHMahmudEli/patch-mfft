"""
Download Pretrained Patch-MFFT Model Checkpoints from Hugging Face Hub.
Public repository: https://huggingface.co/MohsinElis/patch-mfft-tampering-checkpoints
"""

import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "MohsinElis/patch-mfft-tampering-checkpoints"
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "model" / "checkpoints"

MODELS = {
    "tiny": {
        "filename": "tiny.pt",
        "remote_path": "runs/patch_mfft_tiny/checkpoints/best.pt",
        "params": "563K",
        "accuracy": "97.21%",
        "miou": "79.07%",
    },
    "base": {
        "filename": "base.pt",
        "remote_path": "runs/patch_mfft_base/checkpoints/best.pt",
        "params": "3.28M",
        "accuracy": "97.84%",
        "miou": "81.00%",
    },
    "large": {
        "filename": "large.pt",
        "remote_path": "runs/patch_mfft_large/checkpoints/best.pt",
        "params": "14.78M",
        "accuracy": "92.48%",
        "miou": "47.29%",
    },
}


def download_models(target_model: str = "all"):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = MODELS.keys() if target_model == "all" else [target_model]

    print(f"=== Patch-MFFT Checkpoint Downloader ===")
    print(f"Hugging Face Repository: {REPO_ID}")
    print(f"Target Directory:        {CHECKPOINTS_DIR}\n")

    for key in selected:
        info = MODELS[key]
        dest = CHECKPOINTS_DIR / info["filename"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[EXISTS] {info['filename']} already downloaded ({dest.stat().st_size / 1e6:.1f} MB). Skipping.")
            continue

        print(f"[DOWNLOADING] {info['filename']} ({info['params']} params, Acc: {info['accuracy']}, mIoU: {info['miou']})...")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=info["remote_path"],
                repo_type="model",
            )
            shutil.copy2(downloaded, dest)
            print(f"  -> Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)\n")
        except Exception as e:
            print(f"  -> Failed to download {info['filename']}: {e}\n")

    print("Download process completed.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target not in list(MODELS.keys()) + ["all"]:
        print(f"Invalid model target: {target}. Choices: {list(MODELS.keys()) + ['all']}")
        sys.exit(1)
    download_models(target)
