'use client';

import React from 'react';
import { Radio, Activity } from 'lucide-react';

interface FrequencyBreakdownProps {
  contributions: Record<string, number>;
}

export function FrequencyBreakdown({ contributions }: FrequencyBreakdownProps) {
  const bands = [
    {
      key: 'band_1',
      name: 'Low-Frequency (Blow)',
      radius: 'r in [0.0, 0.15]',
      desc: 'Macro illumination, structural contours & geometry',
      color: 'bg-emerald-500',
      textColor: 'text-emerald-400',
    },
    {
      key: 'band_2',
      name: 'Mid-Frequency (Bmid)',
      radius: 'r in (0.15, 0.50]',
      desc: 'Semantic textures & surface patterns',
      color: 'bg-amber-500',
      textColor: 'text-amber-400',
    },
    {
      key: 'band_3',
      name: 'High-Frequency (Bhigh)',
      radius: 'r in (0.50, 1.0]',
      desc: 'Sensor photon shot noise & checkerboard residuals',
      color: 'bg-rose-500',
      textColor: 'text-rose-400',
    },
  ];

  const total = Object.values(contributions).reduce((acc, v) => acc + v, 0) || 1;

  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-2xl p-5 shadow-xl space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-white flex items-center gap-2">
          <Radio className="w-4 h-4 text-indigo-400" />
          Radial Annular Frequency Decomposition
        </h3>
        <span className="text-[11px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">
          2D-FFT Spectral Partition
        </span>
      </div>

      <div className="space-y-3.5">
        {bands.map((b) => {
          const val = contributions[b.key] || 0;
          const pct = Math.min(100, Math.round((val / total) * 100));

          return (
            <div key={b.key} className="space-y-1.5 bg-gray-950/40 p-3 rounded-xl border border-gray-800/60">
              <div className="flex items-center justify-between text-xs">
                <div>
                  <span className={`font-semibold ${b.textColor}`}>{b.name}</span>
                  <span className="text-gray-500 ml-2 font-mono text-[10px]">({b.radius})</span>
                </div>
                <span className="font-mono text-gray-300 font-semibold">{val.toFixed(4)} ({pct}%)</span>
              </div>

              {/* Progress bar */}
              <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full ${b.color} transition-all duration-500 rounded-full`}
                  style={{ width: `${Math.max(4, pct)}%` }}
                />
              </div>

              <p className="text-[11px] text-gray-400">{b.desc}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
