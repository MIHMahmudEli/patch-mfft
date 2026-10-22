# Patch-MFFT Forensics — Web Frontend (Vercel Ready)

Modern Next.js 14 web application for **Patch-MFFT**: Dual-Branch Radial Frequency Transformer for Localized Generative Inpainting, Composite Blackmail Splicing, and Style Tampering Detection (IEEE Transactions on Information Forensics and Security, 2026).

---

## Features

- **3-Way Forensic Triage**: Live probabilistic breakdown across:
  - `[0]` Authentic Optical Camera Capture
  - `[1]` Holistic Generative AI (SDXL, DALL-E 3, Midjourney v6)
  - `[2]` Localized Spliced / Inpainted Tampering (Face swapping, inpainting, blackmail composites)
- **Interactive Tampering Localization Visualizer**:
  - Live side-by-side view and alpha-blended opacity slider overlaying the predicted $384 \times 384$ pixel-level probability mask onto the input photograph.
  - One-click mask download.
- **Radial Annular 2D-FFT Energy Breakdown**:
  - Concentric Low-Band ($\mathcal{B}_{\text{low}}$), Mid-Band ($\mathcal{B}_{\text{mid}}$), and High-Band ($\mathcal{B}_{\text{high}}$) spectral energy distribution.
- **Multi-Model Variant Switching**:
  - Tiny (563K parameters, ~4.2 ms latency)
  - Base (3.28M parameters, 97.84% accuracy, 81.00% mIoU)
  - Large (13.94M parameters, maximum precision)
- **Built-in Presets**: Instant testing for authentic, spliced, and inpainting scenarios without requiring external image files.

---

## Deploying to Vercel

### Method 1: Deploy with Vercel CLI

```bash
cd web
npm install
npm run build   # test production build locally

# Deploy to Vercel
npx vercel
# Follow prompts, set Environment Variable:
# NEXT_PUBLIC_API_URL = https://mohsinelis-patch-mfft-api.hf.space
```

### Method 2: Deploy via Vercel Dashboard (GitHub / Git)

1. Push your repository to GitHub / GitLab.
2. In the [Vercel Dashboard](https://vercel.com/new), select **Import Repository**.
3. Set **Root Directory** to `web`.
4. Under **Environment Variables**, add:
   - `NEXT_PUBLIC_API_URL`: `https://mohsinelis-patch-mfft-api.hf.space`
5. Click **Deploy**.

---

## Local Development

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.
