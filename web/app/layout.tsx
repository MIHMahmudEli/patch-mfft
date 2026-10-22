import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'PatchForensics | Unmasking Generative Forgery at the Pixel Level',
  description:
    'Dual-Branch Frequency-Guided Transformers for Localized Image Tampering and Inpainting Forensics.',
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
