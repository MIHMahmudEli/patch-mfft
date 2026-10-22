'use client';

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon, Zap, AlertCircle, FileCheck2 } from 'lucide-react';

interface UploadZoneProps {
  imagePreview: string | null;
  onImageSelected: (file: File) => void;
  onClear: () => void;
  isLoading: boolean;
  onAnalyze: () => void;
}

export function UploadZone({
  imagePreview,
  onImageSelected,
  onClear,
  isLoading,
  onAnalyze,
}: UploadZoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles && acceptedFiles.length > 0) {
        onImageSelected(acceptedFiles[0]);
      }
    },
    [onImageSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.webp'],
    },
    maxFiles: 1,
    maxSize: 25 * 1024 * 1024,
  });

  // Preset demo generator (creates representative canvas test images)
  const handleLoadSample = (type: 'authentic' | 'spliced' | 'inpainting') => {
    const canvas = document.createElement('canvas');
    canvas.width = 384;
    canvas.height = 384;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (type === 'authentic') {
      // Natural gradient with subtle simulated sensor noise
      const grad = ctx.createLinearGradient(0, 0, 384, 384);
      grad.addColorStop(0, '#3b82f6');
      grad.addColorStop(0.5, '#6366f1');
      grad.addColorStop(1, '#1e1b4b');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 384, 384);
      // Natural noise
      const imgData = ctx.getImageData(0, 0, 384, 384);
      for (let i = 0; i < imgData.data.length; i += 4) {
        const noise = (Math.random() - 0.5) * 12;
        imgData.data[i] = Math.min(255, Math.max(0, imgData.data[i] + noise));
        imgData.data[i + 1] = Math.min(255, Math.max(0, imgData.data[i + 1] + noise));
        imgData.data[i + 2] = Math.min(255, Math.max(0, imgData.data[i + 2] + noise));
      }
      ctx.putImageData(imgData, 0, 0);
    } else if (type === 'spliced') {
      // Background optical scene + pasted composite person box
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, 384, 384);
      // Spliced patch (different frequency distribution)
      ctx.fillStyle = '#f43f5e';
      ctx.fillRect(100, 90, 180, 180);
      ctx.fillStyle = '#fecdd3';
      ctx.font = 'bold 18px sans-serif';
      ctx.fillText('Spliced Region', 120, 180);
    } else {
      // Smooth inpainting fill
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, 384, 384);
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(192, 192, 70, 0, Math.PI * 2);
      ctx.fill();
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `sample_${type}.jpg`, { type: 'image/jpeg' });
        onImageSelected(file);
      }
    }, 'image/jpeg');
  };

  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Upload className="w-4 h-4 text-indigo-400" />
          Image Input
        </h2>
        {/* Sample presets */}
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] text-gray-500 mr-1 hidden sm:inline">Try preset:</span>
          <button
            onClick={() => handleLoadSample('authentic')}
            className="px-2 py-0.5 text-[11px] font-medium bg-gray-800 hover:bg-gray-700 text-gray-300 rounded border border-gray-700/60"
          >
            Authentic
          </button>
          <button
            onClick={() => handleLoadSample('spliced')}
            className="px-2 py-0.5 text-[11px] font-medium bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 rounded border border-rose-800/40"
          >
            Spliced Blackmail
          </button>
          <button
            onClick={() => handleLoadSample('inpainting')}
            className="px-2 py-0.5 text-[11px] font-medium bg-amber-950/60 hover:bg-amber-900/60 text-amber-300 rounded border border-amber-800/40"
          >
            Inpainting
          </button>
        </div>
      </div>

      {!imagePreview ? (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragActive
              ? 'border-indigo-500 bg-indigo-500/10'
              : 'border-gray-700/70 hover:border-gray-500 bg-gray-950/40'
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center gap-3">
            <div className="p-3 bg-gray-800/80 rounded-full text-indigo-400 border border-gray-700">
              <Upload className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-200">
                {isDragActive ? 'Drop image here...' : 'Click to upload or drag & drop'}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                JPEG, PNG, or WEBP up to 25 MB (Resized to 384x384 aspect-preserved)
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-black flex items-center justify-center max-h-[380px]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imagePreview}
              alt="Uploaded forensic sample"
              className="max-h-[360px] w-auto object-contain"
            />
            <button
              onClick={onClear}
              className="absolute top-3 right-3 px-2.5 py-1 text-xs font-medium bg-black/70 hover:bg-black text-gray-300 rounded-md border border-gray-700 backdrop-blur-sm transition-colors"
            >
              Clear
            </button>
          </div>

          <button
            onClick={onAnalyze}
            disabled={isLoading}
            className={`w-full py-3 px-4 rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-lg ${
              isLoading
                ? 'bg-gray-800 text-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-indigo-600 via-indigo-500 to-rose-600 hover:from-indigo-500 hover:to-rose-500 text-white shadow-indigo-500/20 active:scale-[0.99]'
            }`}
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                <span>Decomposing Radial Frequencies & Slicing Tokens...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 text-amber-300" />
                <span>Execute Patch-MFFT Forensics</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
