# OpenVLA-OFT × Wan-WM GRPO POC

This directory is a non-invasive experiment overlay for the pinned RLinf v0.3 commit
`0505431899574619da86f551bad70b71e0ea2177`. It does not modify RLinf itself and it
does not use the existing dirty checkout as a training source.

## Frozen protocol

- Train: frozen action-conditioned Wan and ResNet reward model, KIR enabled, native RLinf v0.3 GRPO.
- Actor: OpenVLA-OFT SFT checkpoint with a new rank-32/alpha-32 LoRA; every non-LoRA parameter remains frozen.
- Budget: 100 completed updates or 72 wall-clock hours from the first formal launch.
- Evaluate: 10 LIBERO-Spatial tasks × 50 canonical states × seeds 1234/1235/1236, exactly 1,500 episodes per model.
- Precision/rendering: training is BF16; adapter merge and both truth evaluations are FP32, with OSMesa fixed for headless rendering.
- Success: the lower endpoint of the paired, task-stratified cluster-bootstrap 95% interval for `FT - Base` is above zero.

Training rollout provenance is fixed: an official local initialization/KIR record
provides the five conditioning frames, the current policy produces each 8×7 action
chunk, frozen Wan imagines the next eight frames, and the frozen local ResNet RM scores
those frames. The compute node neither serves nor calls the GitHub Pages site; Pages
only displays result artifacts exported after the run.

Generate the immutable evaluation protocol once:

```bash
python scripts/create_eval_manifest.py
```

## Required layout on both H20 nodes

Prepare a clean checkout and immutable model snapshots outside this overlay. Set:

```bash
export RLINF_ROOT=/shared/poc/RLinf-v0.3-clean
export POC_PROJECT_ROOT=/shared/poc/rlinf-wan-libero-poc
export OPENVLA_BASE_PATH=/local-cache/Openvla-oft-SFT-libero-spatial-traj1
export WAN_WM_PATH=/local-cache/RLinf-Wan-LIBERO-Spatial
```

The checkout must be detached at the pinned commit. The container must be
`rlinf/rlinf:agentic-rlinf0.3-wan` pinned by digest, not only by tag. Run
`scripts/preflight.py` on each node; it fails if a GPU is occupied or an asset is missing.
When writing `experiment_manifest.yaml`, pass the immutable POC commit, both
Hugging Face revisions and the frozen config to `scripts/record_experiment.py`.
The file is JSON-compatible YAML and intentionally contains no absolute host paths.

Before an 8-GPU run, verify NCCL with a minimal all-reduce inside the pinned image.
On the validated H20 host/image pair, NCCL 2.21.5's NVLS transport fails during the
initial FSDP broadcast with CUDA error 401, while the same 8-GPU all-reduce succeeds
with NVLink/P2P retained and only `NCCL_NVLS_ENABLE=0`. Therefore every canary,
benchmark, and formal training container on that host must export
`NCCL_NVLS_ENABLE=0`; the canary launchers reject a missing or different value.

## Ordered execution

1. Start Ray with `cluster_start.sh` on Node A (`POC_NODE_RANK=0`) and Node B (`POC_NODE_RANK=1`), then run `cluster_check.sh`.
2. Run the six-update benchmark on Node A, Node B, and the joint cluster. Combine the three JSON reports and run `select_topology.py`. The decision is automatic at a 1.5× joint/single threshold.
3. Run `run_full_canary.sh` with all 8 H20 GPUs visible and an exclusive Ray window. The standalone rollout checks one fixed 8-sample group for the full 32 action chunks (256 steps). The real RLinf training canary uses 64 environments (8 GRPO groups), global batch 2048, and micro batch 32. After update 1 it validates TensorBoard scalars and refuses to resume if returns are empty, rewards/advantages are non-finite, or the policy gradient is zero. `POC_CANARY_RECORD_NAME` can pin the standalone rollout record during diagnosis. Formal training is forbidden unless `canary-acceptance.json` says `passed`.
4. Evaluate Base with `run_eval_shard.sh`; merge shards and verify all 1,500 rows before training. The launcher selects OSMesa before importing robosuite and retries native renderer exits 134/139 at most twice for the same pending episode. This avoids the reproducible EGL `mjr_readPixels` abort observed on the validated headless H20 host.
5. Run `run_train.sh` with `POC_NUM_NODES` from `topology_decision.json`. Resume uses the same `budget.json`, so the 72-hour clock never resets.
6. Convert the last complete checkpoint with `export_checkpoint.sh`, then evaluate the merged FT checkpoint using the exact same manifest.
7. Merge Base and FT shards, run `aggregate_results.py`, and copy the generated bundle and media into the static result site's `public/results` directory.

If the binary RM remains constant, run `probe_rm_libero_success.py` first. It
searches the fixed task/state/seed order for a simulator-confirmed `done=True`,
then scores its final frame window and the initial negative frame with the exact
Wan RM preprocessing. Use `select_kir_canary_records.py` to create the deterministic
one-record-per-task manifest for the separate stratified imagined-rollout check.
`POC_CANARY_RECORD_NAME` affects only the standalone diagnostic rollout.
`POC_CANARY_ENV_SEED` controls the real RLinf reset generator and defaults to the
registered training seed `1234`; any override must be recorded with the disposable
canary report and never carried into formal training silently.

The direct evaluator writes each completed episode durably and supports disjoint shards.
Infrastructure exceptions are retried twice and then abort the evaluation without writing a
fake failure. Policy timeouts and invalid/non-finite policy actions are recorded as failures.

## Outputs

Each run keeps its own timestamped directory with preflight reports, asset hashes,
topology decision, canary evidence, TensorBoard logs, checkpoints and the experiment
manifest. Only LoRA/optimizer recovery checkpoints are retained during training; the
final delivery contains the adapter and one merged HF checkpoint.

The result site consumes `summary.json`, `tasks.csv`, `paired_videos.json`,
`imagined_rollout.json`, and the raw `episodes.jsonl`. Until real evaluation data exists,
it displays an explicit waiting state rather than sample performance numbers.
