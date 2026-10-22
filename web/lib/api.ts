import { PredictionResult, ModelsResponse } from './utils';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://mohsinelis-patch-mfft-api.hf.space';

export async function checkHealth(): Promise<{ status: string; models_loaded: string[] }> {
  try {
    const res = await fetch(`${API_BASE}/`, { method: 'GET' });
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.warn('API Health Check failed:', err);
    return { status: 'offline', models_loaded: [] };
  }
}

export async function fetchModels(): Promise<ModelsResponse> {
  try {
    const res = await fetch(`${API_BASE}/models`, { method: 'GET' });
    if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.warn('Using default models config due to fetch error:', err);
    return {
      default: 'large',
      models: [
        { id: 'tiny', loaded: true, params: '563K', description: 'Edge / Mobile fast screening' },
        { id: 'base', loaded: true, params: '3.28M', description: 'Balanced triage and localization' },
        { id: 'large', loaded: true, params: '13.94M', description: 'Maximum precision for inpainting & blackmail' },
      ],
    };
  }
}

export async function runForensicAnalysis(
  imageBlob: Blob,
  modelVariant: string = 'large'
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'sample.jpg');

  const res = await fetch(`${API_BASE}/predict?model=${encodeURIComponent(modelVariant)}`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Forensics API Error (${res.status}): ${errText || res.statusText}`);
  }

  return await res.json();
}
