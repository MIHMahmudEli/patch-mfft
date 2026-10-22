'use client';

import React, { useState, useEffect } from 'react';
import { Header } from '../components/Header';
import { UploadZone } from '../components/UploadZone';
import { ForensicResults } from '../components/ForensicResults';
import { checkHealth, runForensicAnalysis } from '../lib/api';
import { PredictionResult } from '../lib/utils';
import { ShieldCheck, AlertTriangle, Cpu, Layers, Sparkles, BookOpen } from 'lucide-react';

export default function Home() {
  const [selectedModel, setSelectedModel] = useState<string>('base');
  const [apiStatus, setApiStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check backend health on mount
  useEffect(() => {
    async function verifyBackend() {
      const health = await checkHealth();
      setApiStatus(health.status === 'healthy' ? 'online' : 'offline');
    }
    verifyBackend();
    const interval = setInterval(verifyBackend, 25000);
    return () => clearInterval(interval);
  }, []);

  const handleImageSelected = (file: File) => {
    setImageFile(file);
    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleClear = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!imageFile) return;
    setIsLoading(true);
    setError(null);

    try {
      const res = await runForensicAnalysis(imageFile, selectedModel);
      setResult(res);
    } catch (err: any) {
      console.error('Analysis error:', err);
      setError(err.message || 'Forensic analysis failed. Please check the backend connection.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between">
      <Header
        selectedModel={selectedModel}
        onSelectModel={setSelectedModel}
        apiStatus={apiStatus}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full flex-1">
        {/* Hero Section */}
        <div className="text-center max-w-3xl mx-auto mb-10 space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Dual-Branch Spatial-Max Gating (SMG) & FGA Transformer</span>
          </div>

          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            Localized Generative Tampering & Inpainting Forensics
          </h2>

          <p className="text-sm sm:text-base text-gray-400 leading-relaxed">
            Conventional whole-image frequency detectors fail when tampering occupies only 5%–15% of the frame.
            <strong> Patch-MFFT</strong> combines radial 2D-FFT annular filtering with dual-branch peak anomaly gating
            to concurrently execute <strong>3-way frame classification triage</strong> and produce{' '}
            <strong>pixel-wise tampering localization heatmaps</strong>.
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-3 max-w-4xl mx-auto">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Analysis Failed</p>
              <p className="text-xs text-rose-300/80 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Upload & Input */}
          <div className="lg:col-span-5 space-y-6">
            <UploadZone
              imagePreview={imagePreview}
              onImageSelected={handleImageSelected}
              onClear={handleClear}
              isLoading={isLoading}
              onAnalyze={handleAnalyze}
            />

            {/* Architectural Highlights Info Box */}
            <div className="bg-gray-900/40 border border-gray-800/80 rounded-2xl p-4 space-y-3 text-xs text-gray-400">
              <h4 className="font-semibold text-gray-200 flex items-center gap-1.5">
                <BookOpen className="w-4 h-4 text-indigo-400" />
                Forensic Architecture Specifications
              </h4>
              <ul className="space-y-1.5 list-disc list-inside text-gray-400 leading-relaxed">
                <li>
                  <strong className="text-gray-300">Annular Spectral Partitioning:</strong> Low (r &le; 0.15), Mid (0.15 &lt; r &le; 0.50), High (0.50 &lt; r &le; 1.0)
                </li>
                <li>
                  <strong className="text-gray-300">Spatial-Max Gating (SMG):</strong> Unifies holistic scene context (z_avg) with peak localized anomalies (z_max)
                </li>
                <li>
                  <strong className="text-gray-300">Multi-Task Decoder:</strong> 3-way triage (y_hat in R^3) + dense localization (M_hat in [0, 1]^(384x384))
                </li>
              </ul>
            </div>
          </div>

          {/* Right Column: Forensic Results */}
          <div className="lg:col-span-7">
            {result && imagePreview ? (
              <ForensicResults result={result} originalImage={imagePreview} />
            ) : (
              <div className="bg-gray-900/40 border border-gray-800/60 border-dashed rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[460px] text-gray-500 space-y-3">
                <div className="p-4 bg-gray-900 rounded-2xl border border-gray-800 text-gray-600">
                  <Layers className="w-10 h-10 stroke-1" />
                </div>
                <div>
                  <h3 className="text-base font-semibold text-gray-300">Awaiting Image Submission</h3>
                  <p className="text-xs text-gray-500 mt-1 max-w-sm">
                    Upload an image or select a sample preset on the left to trigger real-time radial FFT decomposition,
                    cross-band attention, and pixel-level tampering segmentation.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800/80 bg-gray-950/60 py-6 mt-12 text-xs text-gray-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>© 2026 Patch-MFFT Research Team. All rights reserved.</p>
          <div className="flex items-center gap-4">
            <a
              href="https://huggingface.co/spaces/MohsinElis/patch-mfft-api"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-gray-300 transition-colors"
            >
              HuggingFace Space
            </a>
            <span>•</span>
            <a
              href="https://huggingface.co/MohsinElis/patch-mfft-tampering-checkpoints"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-gray-300 transition-colors"
            >
              Model Checkpoints
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
