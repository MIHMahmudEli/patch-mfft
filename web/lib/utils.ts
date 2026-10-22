import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export type Verdict = 'authentic' | 'full_ai' | 'spliced_tampered';

export interface PredictionResult {
  prediction: Verdict;
  confidence: number;
  tampering_detected: boolean;
  class_probabilities: {
    authentic: number;
    full_ai: number;
    spliced_tampered: number;
  };
  tampering_heatmap_b64: string | null;
  frequency_band_contributions: Record<string, number>;
  processing_time_ms: number;
  model_variant: string;
}

export interface ModelInfo {
  id: string;
  loaded: boolean;
  params: string;
  description: string;
}

export interface ModelsResponse {
  default: string;
  models: ModelInfo[];
}

export function getVerdictConfig(prediction: Verdict) {
  switch (prediction) {
    case 'authentic':
      return {
        label: 'Authentic Capture',
        description: 'Verified optical camera shot. Coherent Bayer demosaicing & sensor photon shot noise.',
        color: 'text-emerald-500',
        bg: 'bg-emerald-500/10',
        border: 'border-emerald-500/30',
        badge: 'bg-emerald-500 text-white',
        status: 'AUTHENTIC',
      };
    case 'full_ai':
      return {
        label: 'Holistic Generative AI',
        description: 'Full-frame synthetic synthesis. Global frequency grid & deconvolution checkerboard harmonics.',
        color: 'text-amber-500',
        bg: 'bg-amber-500/10',
        border: 'border-amber-500/30',
        badge: 'bg-amber-500 text-white',
        status: 'SYNTHETIC AI',
      };
    case 'spliced_tampered':
      return {
        label: 'Spliced / Inpainted Tampering',
        description: 'Localized manipulation detected. Boundary phase incoherence & localized inpainting diffusion traces.',
        color: 'text-rose-500',
        bg: 'bg-rose-500/10',
        border: 'border-rose-500/30',
        badge: 'bg-rose-500 text-white',
        status: 'TAMPERED ALERT',
      };
    default:
      return {
        label: 'Unknown State',
        description: 'Analysis inconclusive.',
        color: 'text-gray-400',
        bg: 'bg-gray-500/10',
        border: 'border-gray-500/30',
        badge: 'bg-gray-500 text-white',
        status: 'UNKNOWN',
      };
  }
}
