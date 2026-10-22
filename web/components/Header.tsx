'use client';

import React from 'react';
import { ShieldCheck, Cpu, Activity, ExternalLink, Github, Database } from 'lucide-react';

interface HeaderProps {
  selectedModel: string;
  onSelectModel: (variant: string) => void;
  apiStatus: 'online' | 'offline' | 'checking';
}

export function Header({ selectedModel, onSelectModel, apiStatus }: HeaderProps) {
  const variants = [
    { id: 'tiny', label: 'Tiny', params: '563K', desc: 'Edge profile (~4 ms)' },
    { id: 'base', label: 'Base', params: '3.28M', desc: '97.84% Acc, 81.0% mIoU (Default)' },
    { id: 'large', label: 'Large', params: '13.94M', desc: 'Deep model profile' },
  ];

  return (
    <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-tr from-rose-600 via-indigo-600 to-blue-500 rounded-xl shadow-lg shadow-indigo-500/20">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white font-sans">
                PatchForensics <span className="text-indigo-400 font-semibold text-xs tracking-wider uppercase bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded-full">AI</span>
              </h1>
            </div>
            <p className="text-xs text-gray-400 hidden sm:block">
              Dual-Branch Radial Frequency Transformer for Localized Inpainting & Splicing
            </p>
          </div>
        </div>

        {/* Variant Selector + Status */}
        <div className="flex items-center gap-3 flex-wrap justify-center">
          {/* Variant Selector */}
          <div className="flex items-center bg-gray-900 border border-gray-800 p-1 rounded-lg">
            {variants.map((v) => {
              const active = selectedModel === v.id;
              return (
                <button
                  key={v.id}
                  onClick={() => onSelectModel(v.id)}
                  title={v.desc}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-all flex items-center gap-1.5 ${
                    active
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  <span>{v.label}</span>
                  <span className={`text-[10px] ${active ? 'text-indigo-200' : 'text-gray-500'}`}>
                    ({v.params})
                  </span>
                </button>
              );
            })}
          </div>

          {/* API Health */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-gray-900/60 border border-gray-800 text-xs text-gray-300">
            <span
              className={`w-2 h-2 rounded-full ${
                apiStatus === 'online'
                  ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50 animate-pulse'
                  : apiStatus === 'checking'
                  ? 'bg-amber-500'
                  : 'bg-rose-500'
              }`}
            />
            <span className="text-[11px] font-mono capitalize">{apiStatus}</span>
          </div>

          {/* Hugging Face Space Link */}
          <a
            href="https://huggingface.co/spaces/MohsinElis/patch-mfft-api"
            target="_blank"
            rel="noopener noreferrer"
            className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800 transition-colors"
            title="HF Space API"
          >
            <Database className="w-4 h-4" />
          </a>
        </div>
      </div>
    </header>
  );
}
