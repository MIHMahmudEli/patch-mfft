import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Patch-MFFT | Localized Tampering & Inpainting Forensics',
  description:
    'Dual-Branch Radial Frequency Transformer for Localized Generative Inpainting, Composite Blackmail Splicing, and Style Tampering Detection (IEEE TIFS 2026).',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased selection:bg-indigo-500/30 selection:text-indigo-200">
        {children}
      </body>
    </html>
  );
}
