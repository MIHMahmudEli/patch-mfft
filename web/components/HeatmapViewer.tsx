'use client';

import React, { useState } from 'react';
import { Eye, Layers, Sliders, Download } from 'lucide-react';

interface HeatmapViewerProps {
  originalImage: string;
  heatmapBase64: string | null;
  prediction: string;
}

export function HeatmapViewer({ originalImage, heatmapBase64, prediction }: HeatmapViewerProps) {
  const [opacity, setOpacity] = useState<number>(0.65);
  const [viewMode, setViewMode] = useState<'overlay' | 'side-by-side'>('overlay');

  if (!heatmapBase64) {
    return (
      <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-5 text-center text-gray-500 text-sm">
        Tampering localization mask not available for this variant.
      </div>
    );
  }

  const maskDataUrl = `data:image/png;base64,${heatmapBase64}`;

  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-5 shadow-xl space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-rose-400" />
          <h3 className="text-base font-semibold text-white">Dense Tampering Localization Mask</h3>
          <span className="text-[11px] px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20">
            384 × 384 Pixel Heatmap
          </span>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode(viewMode === 'overlay' ? 'side-by-side' : 'overlay')}
            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 transition-colors flex items-center gap-1.5"
          >
            <Eye className="w-3.5 h-3.5" />
            <span>{viewMode === 'overlay' ? 'Side-by-Side View' : 'Blend Overlay View'}</span>
          </button>

          <a
            href={maskDataUrl}
            download="patch_mfft_tampering_mask.png"
            className="p-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 transition-colors"
            title="Download predicted mask"
          >
            <Download className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>

      {viewMode === 'overlay' ? (
        <div className="space-y-3">
          {/* Composite Overlay Canvas */}
          <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-black flex items-center justify-center min-h-[340px]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={originalImage}
              alt="Original Image"
              className="max-h-[360px] w-auto object-contain block"
            />
            {/* Heatmap overlay */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={maskDataUrl}
              alt="Tampering Heatmap Mask"
              style={{
                opacity: opacity,
                mixBlendMode: 'screen',
                filter: 'drop-shadow(0 0 10px rgba(239, 68, 68, 0.8)) hue-rotate(-20deg) saturate(200%)',
              }}
              className="absolute inset-0 m-auto max-h-[360px] w-auto object-contain pointer-events-none"
            />
          </div>

          {/* Slider */}
          <div className="flex items-center gap-3 px-2 pt-1">
            <Sliders className="w-4 h-4 text-gray-400" />
            <span className="text-xs text-gray-400 font-medium">Mask Intensity:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="flex-1 accent-rose-500 h-1.5 bg-gray-800 rounded-lg cursor-pointer"
            />
            <span className="text-xs font-mono text-gray-300 w-10 text-right">
              {Math.round(opacity * 100)}%
            </span>
          </div>
        </div>
      ) : (
        /* Side by Side View */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <span className="text-xs font-medium text-gray-400">Original Capture:</span>
            <div className="rounded-xl overflow-hidden border border-gray-800 bg-black flex items-center justify-center p-1 min-h-[260px]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={originalImage}
                alt="Original"
                className="max-h-[260px] w-auto object-contain"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <span className="text-xs font-medium text-rose-400">Predicted Tampering Probability Mask:</span>
            <div className="rounded-xl overflow-hidden border border-rose-900/50 bg-black flex items-center justify-center p-1 min-h-[260px]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={maskDataUrl}
                alt="Tampering Probability Mask"
                className="max-h-[260px] w-auto object-contain invert brightness-125 contrast-150"
              />
            </div>
          </div>
        </div>
      )}

      <p className="text-xs text-gray-500 italic text-center">
        Bright highlights denote detected localized diffusion inpainting contours and two-person composite splicing seams.
      </p>
    </div>
  );
}
