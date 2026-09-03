import type { NextConfig } from 'next';

const pagesBasePath = (process.env.PAGES_BASE_PATH ?? '').replace(/\/$/, '');

const nextConfig: NextConfig = {
  output: 'export',
  images: { unoptimized: true },
  assetPrefix: pagesBasePath || undefined,
};

export default nextConfig;
