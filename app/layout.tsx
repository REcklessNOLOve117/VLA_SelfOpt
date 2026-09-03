import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'OpenVLA-OFT × Wan-WM GRPO | LIBERO-Spatial POC',
  description: 'Base OpenVLA-OFT 与 Wan 世界模型 GRPO LoRA 后训练模型的 LIBERO-Spatial 配对评测。',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
