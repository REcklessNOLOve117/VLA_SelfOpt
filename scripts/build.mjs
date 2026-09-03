import { existsSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

const projectRoot = process.cwd();
const dist = join(projectRoot, 'dist');
const staticIndex = join(dist, 'client', 'index.html');
rmSync(dist, { recursive: true, force: true });

const result = spawnSync(
  process.execPath,
  [join(projectRoot, 'node_modules', 'vinext', 'dist', 'cli.js'), 'build'],
  { cwd: projectRoot, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 },
);
process.stdout.write(result.stdout ?? '');
process.stderr.write(result.stderr ?? '');

const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
const freshStaticExport = output.includes('Build complete.') && output.includes("Pre-rendering all routes (output: 'export')") && existsSync(staticIndex);
if (result.status !== 0 && freshStaticExport && output.includes('UV_HANDLE_CLOSING')) {
  console.warn('[build] vinext completed the fresh static export; ignored its Windows-only shutdown assertion.');
  process.exit(0);
}
process.exit(result.status ?? 1);
