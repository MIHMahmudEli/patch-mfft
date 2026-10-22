# Patch-MFFT: Dual-Branch Frequency-Guided Transformers for Localized Image Tampering and Inpainting Forensics

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space%20Demo-yellow)](https://huggingface.co/spaces/MohsinElis/patch-mfft-api)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-Hub-purple)](https://huggingface.co/MohsinElis/patch-mfft-tampering-checkpoints)

Official open-source implementation and evaluation suite for **"Patch-MFFT: Dual-Branch Frequency-Guided Transformers for Localized Image Tampering and Inpainting Forensics"**.

---

## 🌟 Key Highlights

- **Multi-Band Radial Spectral Decomposition**: Decomposes 2D Discrete Cosine Transform (DCT) log-magnitude spectra into radial concentric forensic frequency bands (*Low*, *Mid*, *High*), capturing phase artifacts and generative footprint anomalies.
- **Frequency-Guided Attention (FGA)**: Cross-band attention blocks that dynamically cross-attend high-frequency edge gradients to low-frequency semantic tokens, detecting boundary discontinuities invisible to standard RGB vision transformers.
- **Dual-Branch Multi-Task Head**: Simultaneously outputs global image authentic/tampered classification probabilities and high-resolution $384 \times 384$ dense tampering localization heatmaps.
- **Superior Tampering Localization**: Outperforms conventional vision backbones (CLIP, ViT-B/16, Swin-T, ResNet-50) by **+14.5% to +43.2% accuracy** and delivers **81.00% mIoU** on challenging generative composite and inpainting forgeries.

---

## 📊 Pretrained Models & Benchmark Performance

All models were evaluated on the held-out multi-generator tampering test set ($N = 18{,}000$ images) across diverse generative families (Stable Diffusion, Midjourney, StyleGAN, ProGAN, Deepfake composite frames, and Adobe Firefly inpainting).

| Model Variant | Parameters | Classification Acc | Macro F1 | Localization mIoU | Checkpoint Link |
|---|---|---|---|---|---|
| **Patch-MFFT Tiny** | **563 K** | 97.21% | 97.21% | 79.07% | [Download (8.7 MB)](https://huggingface.co/MohsinElis/patch-mfft-tampering-checkpoints/resolve/main/runs/patch_mfft_tiny/checkpoints/best.pt) |
| **Patch-MFFT Base** *(Recommended)* | **3.28 M** | **97.84%** | **97.85%** | **81.00%** | [Download (41.4 MB)](https://huggingface.co/MohsinElis/patch-mfft-tampering-checkpoints/resolve/main/runs/patch_mfft_base/checkpoints/best.pt) |
| **Patch-MFFT Large** | **14.78 M** | 92.48% | 92.48% | 47.29% | [Download (169 MB)](https://huggingface.co/MohsinElis/patch-mfft-tampering-checkpoints/resolve/main/runs/patch_mfft_large/checkpoints/best.pt) |

### Baseline Comparison

| Architecture | Parameters | Test Accuracy | Macro F1 | Localization Support |
|---|---|---|---|---|
| ResNet-18 | 11.2 M | 95.83% | 95.83% | ❌ (Classifier only) |
| SimpleCNN | 0.42 M | 93.24% | 93.25% | ❌ (Classifier only) |
| EfficientNet-B0 | 4.0 M | 90.40% | 90.41% | ❌ (Classifier only) |
| ViT-B/16 | 86.1 M | 83.28% | 83.12% | ❌ (Classifier only) |
| FreqDetect | 14.7 K | 82.96% | 82.93% | ❌ (Classifier only) |
| ResNet-50 | 23.5 M | 80.36% | 80.00% | ❌ (Classifier only) |
| CLIP (ViT-B/32) | 87.9 M | 74.23% | 73.67% | ❌ (Classifier only) |
| Swin-T | 27.5 M | 54.58% | 44.59% | ❌ (Classifier only) |
| **Patch-MFFT Base (Ours)** | **3.28 M** | **97.84%** | **97.85%** | **✅ (81.00% mIoU)** |

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/MIHMahmudEli/patch-mfft.git
cd patch-mfft
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

Download all models or a specific variant from the public Hugging Face Hub:

```bash
# Download all variants (Tiny, Base, Large)
python scripts/download_checkpoints.py all

# Or download only the recommended Base model
python scripts/download_checkpoints.py base
```

---

## 💻 Usage

### Command-Line Inference (CLI)

Run tampering detection and generate a high-resolution heatmap overlay:

```bash
python predict.py --image path/to/sample.jpg --model base --output output_heatmap.png
```

Sample CLI output:
```text
=== Patch-MFFT Forensic Inference Engine ===
Device:           cuda
Model Variant:    patch_mfft_base
[Model] Successfully loaded weights from: model/checkpoints/base.pt

--- Forensic Analysis Results ---
Decision:         TAMPERED / FORGERY DETECTED
Authentic Prob:   1.24%
Tampered Prob:    98.76%
Peak Mask Value:  94.31%

Saved localization heatmap overlay to: output_heatmap.png
```

---

### REST API Service (FastAPI)

Launch the production REST API with interactive Swagger documentation:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Interactive Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Predict endpoint: `POST /predict?model=base`

Example cURL request:
```bash
curl -X POST "http://localhost:8000/predict?model=base" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.jpg"
```

---

### Interactive Web Frontend (Next.js 14)

Launch the forensic analyst dashboard with interactive before/after image comparison slider and frequency breakdown visualizer:

```bash
cd web
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

### Docker Deployment

Deploy both API and Web services using Docker Compose:

```bash
docker compose up -d
```

---

## 🌐 Live Cloud Demo

- **Hugging Face Space**: [https://huggingface.co/spaces/MohsinElis/patch-mfft-api](https://huggingface.co/spaces/MohsinElis/patch-mfft-api)
- **Public API Swagger**: [https://mohsinelis-patch-mfft-api.hf.space/docs](https://mohsinelis-patch-mfft-api.hf.space/docs)

---

## 📜 Citation

If you find Patch-MFFT useful in your research or application, please cite our manuscript:

```bibtex
@article{eli2026patchmfft,
  title={Patch-MFFT: Dual-Branch Frequency-Guided Transformers for Localized Image Tampering and Inpainting Forensics},
  author={Eli, Mohsin and Research Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
