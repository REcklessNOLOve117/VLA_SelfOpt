import { existsSync, renameSync, rmdirSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';

const projectRoot = process.cwd();
const clientRoot = resolve(projectRoot, 'dist', 'client');
const basePath = (process.env.PAGES_BASE_PATH ?? '').trim().replace(/^\/+|\/+$/g, '');

if (!basePath) process.exit(0);
if (basePath.includes('..') || basePath.includes('\\')) {
  throw new Error(`Unsafe GitHub Pages base path: ${basePath}`);
}

const prefixedRoot = resolve(clientRoot, basePath);
const prefixedAssets = join(prefixedRoot, '_next');
const rootAssets = join(clientRoot, '_next');
const withinClient = (path) => {
  const location = relative(clientRoot, path);
  return location && !location.startsWith(`..${sep}`) && location !== '..';
};

if (!withinClient(prefixedRoot) || !withinClient(prefixedAssets) || !withinClient(rootAssets)) {
  throw new Error('Refusing to move assets outside dist/client');
}
if (!existsSync(prefixedAssets)) {
  throw new Error(`Expected prefixed assets at ${prefixedAssets}`);
}
if (existsSync(rootAssets)) {
  throw new Error(`Unexpected duplicate asset directory at ${rootAssets}`);
}

renameSync(prefixedAssets, rootAssets);
rmdirSync(prefixedRoot);
