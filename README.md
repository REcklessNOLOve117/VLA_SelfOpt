# OpenVLA-OFT × Wan-WM GRPO POC

This repository contains the static, data-driven result page and the reproducible
LIBERO-Spatial experiment harness in `experiment/`. There is no inference API and
no live GPU dependency in the published site.

GRPO rollouts are generated online inside the frozen action-conditioned Wan world
model. Resets come from the official Wan bundle's local `dataset/` records (including
KIR context); actions come from the current OpenVLA-OFT policy; rewards come from the
local frozen `resnet_rm.pth`. GitHub Pages is only a static consumer of exported result
files and is never part of training, evaluation, or server control.

The page reads these files from `public/results`:

- `summary.json` — Base/FT aggregate and per-task success rates.
- `paired_videos.json` — the 20 pre-registered comparison keys and published videos.
- `imagined_rollout.json` — five conditioning frames, an 8×7 action chunk, eight imagined frames, and reward values.
- `episodes.jsonl` — all 3,000 raw evaluation rows after the experiment completes.

Before real results are copied in, the checked-in files deliberately carry
`status: awaiting_data` and show no fabricated success rates.

```bash
npm run dev
npm test
npm run typecheck
npm run lint
npm run build
```

The production build is a static export in `dist/client`, matching
the GitHub Pages workflow.

## GitHub Pages

Pushes to `main` run tests, type checking, linting, and the static build before
deploying with the official GitHub Pages Actions. The workflow supplies the
repository base path at build time, so assets also work for project Pages URLs.

In the repository settings, select **GitHub Actions** as the Pages source if it is
not selected automatically. The site intentionally remains in `awaiting_data`
state until the real Base/FT evaluation artifacts are generated.

## Experiment harness

See [`experiment/README.md`](experiment/README.md) for the pinned RLinf checkout,
canary, topology benchmark, GRPO run, evaluation, statistics, and result export
commands.
