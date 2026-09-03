import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const readJson = (name) => JSON.parse(readFileSync(new URL(`../public/results/${name}`, import.meta.url), 'utf8'));

test('summary exposes the frozen protocol without fabricated metrics', () => {
  const summary = readJson('summary.json');
  assert.equal(summary.protocol.suite, 'LIBERO-Spatial');
  assert.equal(summary.protocol.tasks, 10);
  assert.equal(summary.protocol.states_per_task, 50);
  assert.deepEqual(summary.protocol.sampling_seeds, [1234, 1235, 1236]);
  assert.equal(summary.protocol.episodes_per_model, 1500);
  assert.equal(summary.tasks.length, 10);
  if (summary.status === 'awaiting_data') {
    assert.equal(summary.overall.base_sr, null);
    assert.equal(summary.overall.ft_sr, null);
    assert.equal(summary.overall.delta, null);
  }
});

test('paired-video registry contains the twenty pre-registered keys', () => {
  const registry = readJson('paired_videos.json');
  assert.equal(registry.pairs.length, 20);
  const selections = new Set(registry.pairs.map((pair) => `${pair.init_state_id}/${pair.sampling_seed}`));
  assert.deepEqual(selections, new Set(['0/1234', '25/1235']));
});

test('imagined rollout is pinned to the pre-registered canary key', () => {
  const rollout = readJson('imagined_rollout.json');
  assert.equal(rollout.episode_key, 'task-00__state-00__seed-1234');
  assert.ok(Array.isArray(rollout.condition_frames));
  assert.ok(Array.isArray(rollout.generated_frames));
});

test('GitHub Pages deployment keeps JSON and media under the repository base path', () => {
  const source = readFileSync(new URL('../app/results-dashboard.tsx', import.meta.url), 'utf8');
  const config = readFileSync(new URL('../next.config.ts', import.meta.url), 'utf8');
  const workflow = readFileSync(new URL('../.github/workflows/pages.yml', import.meta.url), 'utf8');

  assert.match(source, /NEXT_PUBLIC_BASE_PATH/);
  assert.match(source, /fetch\(assetUrl\('\/results\/summary\.json'\)/);
  assert.match(config, /PAGES_BASE_PATH/);
  assert.match(config, /assetPrefix/);
  assert.match(workflow, /scripts\/prepare-pages\.mjs/);
  assert.match(workflow, /actions\/upload-pages-artifact@v4/);
  assert.match(workflow, /actions\/deploy-pages@v4/);
});
