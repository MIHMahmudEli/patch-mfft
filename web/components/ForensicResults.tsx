'use client';

import React from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Cpu,
  Clock,
  Sparkles,
  BarChart3,
  Layers,
} from 'lucide-react';
import { PredictionResult, getVerdictConfig, formatPercent } from '../lib/utils';
import { HeatmapViewer } from './HeatmapViewer';
import { FrequencyBreakdown } from './FrequencyBreakdown';

interface ForensicResultsProps {
  result: PredictionResult;
  originalImage: string;
}

export function ForensicResults({ result, originalImage }: ForensicResultsProps) {
  const config = getVerdictConfig(result.prediction);
  const probs = result.class_probabilities;

  return (
    <div className="space-y-6">
      {/* Main Verdict Card */}
      <div
        className={`rounded-2xl p-6 border ${config.border} ${config.bg} shadow-2xl backdrop-blur-md transition-all duration-300`}
      >
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div
              className={`p-3.5 rounded-2xl ${
                result.prediction === 'authentic'
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  : result.prediction === 'full_ai'
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  : 'bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse'
              }`}
            >
              {result.prediction === 'authentic' ? (
                <ShieldCheck className="w-8 h-8" />
              ) : (
                <ShieldAlert className="w-8 h-8" />
              )}
            </div>

            <div>
              <div className="flex items-center gap-2">
                <span
                  className={`px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider rounded-md ${config.badge}`}
                >
                  {config.status}
                </span>
                <span className="text-xs text-gray-400 font-mono">
                  Variant: <span className="text-white font-semibold capitalize">{result.model_variant}</span>
                </span>
              </div>
              <h2 className={`text-2xl font-extrabold tracking-tight mt-1 ${config.color}`}>
                {config.label}
              </h2>
              <p className="text-xs text-gray-300 mt-0.5 max-w-xl">{config.description}</p>
            </div>
          </div>

          {/* Confidence Meter */}
          <div className="text-left md:text-right bg-gray-950/60 p-3.5 rounded-xl border border-gray-800/80 min-w-[140px]">
            <span className="text-[11px] uppercase tracking-wider text-gray-400 font-semibold block">
              Confidence
            </span>
            <span className={`text-3xl font-black font-mono ${config.color}`}>
              {formatPercent(result.confidence)}
            </span>
          </div>
        </div>

        {/* 3-Class Probabilities Grid */}
        <div className="mt-6 pt-5 border-t border-gray-800/60 grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* Authentic */}
          <div className="bg-gray-950/50 p-3 rounded-xl border border-emerald-900/30 space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="text-emerald-400 font-medium">Authentic Camera</span>
              <span className="font-mono text-gray-200 font-semibold">{formatPercent(probs.authentic)}</span>
            </div>
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(3, probs.authentic * 100)}%` }}
              />
            </div>
          </div>

          {/* Full AI */}
          <div className="bg-gray-950/50 p-3 rounded-xl border border-amber-900/30 space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="text-amber-400 font-medium">Holistic Generative AI</span>
              <span className="font-mono text-gray-200 font-semibold">{formatPercent(probs.full_ai)}</span>
            </div>
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-amber-500 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(3, probs.full_ai * 100)}%` }}
              />
            </div>
          </div>

          {/* Spliced / Tampered */}
          <div className="bg-gray-950/50 p-3 rounded-xl border border-rose-900/30 space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="text-rose-400 font-medium">Spliced / Inpainted</span>
              <span className="font-mono text-gray-200 font-semibold">
                {formatPercent(probs.spliced_tampered)}
              </span>
            </div>
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-rose-500 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(3, probs.spliced_tampered * 100)}%` }}
              />
            </div>
          </div>
        </div>

        {/* Telemetry pill row */}
        <div className="mt-4 flex items-center justify-between text-[11px] text-gray-400 font-mono flex-wrap gap-2">
          <div className="flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-gray-500" />
            <span>Inference Latency: <strong className="text-gray-200">{result.processing_time_ms} ms</strong></span>
          </div>
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Spatial-Max Gating (SMG) & FGA Active</span>
          </div>
        </div>
      </div>

      {/* Heatmap Localization Mask Visualizer */}
      <HeatmapViewer
        originalImage={originalImage}
        heatmapBase64={result.tampering_heatmap_b64}
        prediction={result.prediction}
      />

      {/* Radial Frequency Decomposition */}
      <FrequencyBreakdown contributions={result.frequency_band_contributions} />
    </div>
  );
}
