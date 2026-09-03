# OpenVLA-OFT × Wan-WM GRPO POC

This directory is a non-invasive experiment overlay for the pinned RLinf v0.3 commit
`0505431899574619da86f551bad70b71e0ea2177`. It does not modify RLinf itself and it
does not use the existing dirty checkout as a training source.

## Frozen protocol

- Train: frozen action-conditioned Wan and ResNet reward model, KIR enabled, native RLinf v0.3 GRPO.
- Actor: OpenVLA-OFT SFT checkpoint with a new rank-32/alpha-32 LoRA; every non-LoRA parameter remains frozen.
- Budget: 100 completed updates or 72 wall-clock hours from the first formal launch.
- Evaluate: 10 LIBERO-Spatial tasks × 50 canonical states × seeds 1234/1235/1236, exactly 1,500 episodes per model.
- Success: the lower endpoint of the paired, task-stratified cluster-bootstrap 95% interval for `FT - Base` is above zero.

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

## Ordered execution

1. Start Ray with `cluster_start.sh` on Node A (`POC_NODE_RANK=0`) and Node B (`POC_NODE_RANK=1`), then run `cluster_check.sh`.
2. Run the six-update benchmark on Node A, Node B, and the joint cluster. Combine the three JSON reports and run `select_topology.py`. The decision is automatic at a 1.5× joint/single threshold.
3. Run `run_full_canary.sh` on an exclusive cluster. Formal training is forbidden unless `canary-acceptance.json` says `passed`.
4. Evaluate Base with `run_eval_shard.sh`; merge shards and verify all 1,500 rows before training.
5. Run `run_train.sh` with `POC_NUM_NODES` from `topology_decision.json`. Resume uses the same `budget.json`, so the 72-hour clock never resets.
6. Convert the last complete checkpoint with `export_checkpoint.sh`, then evaluate the merged FT checkpoint using the exact same manifest.
7. Merge Base and FT shards, run `aggregate_results.py`, and copy the generated bundle and media into the static result site's `public/results` directory.

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
