# Handoff: LingBot single-seed CMBench runs on H200

Goal: seed 2, all 116 CMBench tasks, for the methods below. Each run produces one
`continuation.mp4` per task, the official OWL/SAM/DINO score, seqPR and FPS, and a
summary table with a 95% bootstrap CI and the top-5-frame DINO.

## 1. Where everything is

| What | Where |
|---|---|
| Code | https://github.com/Aoraku/CMBench (`main`) |
| Checkpoint | HF `robbyant/lingbot-world-v2-14b-causal-fast` (81 GB) |
| Benchmark | HF dataset `Aoraku/CMBench` (58 videos, 116 tasks, `metadata.jsonl`) |
| Evaluator models | HF `google/owlvit-base-patch32`, `facebook/sam2.1-hiera-large`, `facebook/dinov2-base` |
| Evaluator code | `third_party/cmbench_evaluator/eval_context_memory_relevance.py` (bundled) |
| Head map (DeCoPrune-HS, ForcingKV) | `configs/head_maps/lingbot_heads_q0_sink4_recent4_thr08_cal15_v2.json` (picked up automatically) |
| Method defaults (budgets) | `src/cmbench_rebuild/methods/<method>.py`, `SPEC` defaults |
| New-method implementation notes and deviations from the papers | `docs/rebuttal_methods.md` |
| Offline PR check | `tools/simulate_pr.py` |
| One-command runner | `tools/run_benchmark.py` |

## 2. Setup (once)

```bash
git clone https://github.com/Aoraku/CMBench && cd CMBench
bash tools/setup_env.sh          # .venv (generation) + .venv-eval (evaluator, transformers==5.12.1)
bash tools/download_assets.sh    # models/ and data/CMBench; copies config.json into transformers/
```

`ffmpeg` and `ffprobe` must be on `PATH`. If the cluster has no internet, download on a
machine that does and copy `models/` and `data/` across. H200 runs flash-attn 2 normally;
on Blackwell GPUs (B200/B300) flash-attn 2 has no kernels and the code falls back to
PyTorch SDPA automatically.

## 3. Methods and names

| Paper name | `--methods` id | Budget (default) | Expected seqPR |
|---|---|---|---|
| FullKV | `fullkv` | — | 0 |
| Streaming | `streaming` | sink 4 + local 8 frames | ≈0.94 |
| DummyForcing | `dummy_forcing` | 4/8/4 frames per head group | ≈0.98 |
| ForcingKV | `forcingkv` | K=256 patches, 32 chunks | ≈0.81 |
| TempDiff | `patchification` | 1,657 blocks | ≈0.86 |
| DeCoPrune | `decoprune` | probe step 2, γ=0.10 (set by the runner) | ≈0.854 |
| DeCoPrune-HS | `decoprune_hs` | same + head map | ≈0.862 |
| Random (matched budget) | `random` | DeCoPrune's per-chunk count | ≈0.854 |
| ContextFree (new) | `contextfree` | DeCoPrune's per-chunk count, context-free score | = DeCoPrune |
| SnapKV (new) | `snapkv` | `keep_ratio=0.10` | 0.857 |
| FocusedForcing (new) | `focused_forcing` | `frame_budget=36` | 0.856 |
| WorldKV (new) | `worldkv` | `retrieved_chunks=14`, camera retrieval | 0.854 |
| FutureForcing (new) | `future_forcing` | `keep_ratio=0.10` | 0.857 |
| DensityKV (new) | `densitykv` | **being recalibrated — start this method last** | full-context check gave 0.92 with `growth_limit=2.0` (the bank never fills); a new default is being pushed |
| NoContext (Table 3 lower bound, new) — **please include** | `nocontext` | drops every context token (incl. sink) when generation starts; generated chunks kept | 0.976 (measured) |

All defaults are already calibrated: no `--method-params` are needed. To change a
budget, pass e.g. `--method-params '{"snapkv": {"keep_ratio": 0.12}}'` and check the
resulting PR first with `python tools/simulate_pr.py --methods snapkv --params '{"snapkv": {"keep_ratio": 0.12}}'`.
RoPE re-indexing (`all_bands`) is applied to every method, as in Table 1.

## 4. Running on H200

**One GPU per job is the simplest and is verified for every method** (H100, H200, B300).
Multi-GPU sequence parallel (`--gpus 0,1`) is also verified for all 14 methods (2-GPU
smoke on H100 passed for DeCoPrune and all six new methods). On a 141 GB H200 set
`--gpu-memory-gib 130` so the KV cache stays on-device; KV that does not fit is kept
in host memory automatically (slower but exact).

Run several single-GPU jobs side by side. Each job can take a list of methods and
they run one after another. Jobs that share an `--output-root` must not start at the
exact same second, because the first one writes `inputs/requests.jsonl`. Start them
about 30 s apart, or create the file once with a short run beforehand.

```bash
R=runs/cmbench_seed2
COMMON="--seeds 2 --gpu-memory-gib 130 --checkpoint models/lingbot-world-v2-14b-causal-fast \
  --benchmark-root data/CMBench --model-root models --output-root $R --eval-python .venv-eval/bin/python"
# example split for 8 GPUs: the 7 extra methods (Random + 6 new), heaviest first
.venv/bin/python tools/run_benchmark.py --methods focused_forcing          --gpus 0 $COMMON &
sleep 30
.venv/bin/python tools/run_benchmark.py --methods worldkv                  --gpus 1 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods densitykv                --gpus 2 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods contextfree              --gpus 3 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods future_forcing           --gpus 4 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods snapkv                   --gpus 5 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods random                   --gpus 6 $COMMON &
```

To split one method across GPUs, give each job a disjoint `--cases ...` list. The
task ids are the `task_id` field of `data/CMBench/metadata.jsonl`.
Everything is resumable: re-running the same command skips tasks whose
`continuation.mp4` and `metadata.json` exist. A task that failed part-way is moved to
`<task>/seed2.failed-<time>` and regenerated.

**Timing reference** (one task, full 60 s context, single GPU, measured):

| Method | H200 (1 GPU) | B300 (1 GPU) |
|---|---|---|
| DeCoPrune | — | 5.5 min |
| FullKV | — | ~5 min (generation 293 s) |
| FutureForcing | 11–12 min | |
| ContextFree | 15 min (extra no-cache probe per chunk) | 6 min |
| WorldKV | 22 min (retrieval gathers stored chunks every call) | |

FocusedForcing keeps the full history (sparse attention, no eviction), so its KV is
FullKV-sized; expect it to be the slowest together with WorldKV. For the 116-task run,
split these two over several GPUs with disjoint `--cases` lists.

## 5. Outputs

```
$R/inputs/requests.jsonl                     frozen requests (116 tasks x seeds)
$R/configs/<method>.json                     exact config used per method
$R/runs/<method>/<task>/seed2/               continuation.mp4, metadata.json, seq_pr_metrics.json, events.rank0.jsonl
$R/runs/<method>/<task>/seed2.generation.log per-task log (check here first when a task fails)
$R/eval/<method>/seed2/done/<task>.json      official OWL/SAM/DINO result per task
$R/eval/<method>/seed2/batch*/per_video/...  per-frame scores (used for top-5-frame DINO)
$R/progress.jsonl                            one line per task attempt (status, seconds)
$R/summary/table.md, per_case.csv            final table: DINO, 95% CI, top-5 DINO, seqPR, FPS
```

The summary is rewritten at the end of every invocation; to rebuild it at any
time, rerun the same command (finished tasks are skipped, so it only re-summarizes).

## 6. Sanity checks after the first few tasks

1. `seqPR` in `$R/runs/<method>/<task>/seed2/seq_pr_metrics.json` is close to the expected value above. DensityKV in particular is content dependent; if it drifts far from ≈0.855, tune `max_entries`.
2. For new methods, `events.rank0.jsonl` contains `rebuttal_chunk_finalized` events (WorldKV, SnapKV, FutureForcing, DensityKV) or `consistency_physical_compaction` (ContextFree, Random).
3. WorldKV: `result.json` → `generation.online_policy.rebuttal.visible_subset.retrieval.mode` shows `camera` (pose retrieval) or `query` (fallback when the context camera could not be estimated); both are expected.
4. NoContext is cheap (only one chunk of cache) and is the reference that shows how much a method gains from remembered context; on our two smoke tasks it scored 0.796 / 0.570 DINO, close to DeCoPrune, so the full 116-task comparison against it matters.
5. Reference numbers from our smoke test (2 tasks, seed 2, B300): FullKV DINO 0.924 / 0.825 and DeCoPrune 0.798 / 0.609 on `syn_001_reappear_01` / `_02`; DeCoPrune seqPR 0.863 / 0.866.

## 7. Known pitfalls (already fixed in `main`)

- Missing `transformers/config.json` in the HF checkpoint: `tools/download_assets.sh` copies it.
- Evaluator failing on long target labels: OWL-ViT text is truncated to 16 tokens (`5608fa8`).
- Slow CPU T5 prompt encoding under torchrun (`OMP_NUM_THREADS=1`): the runner sets 16.
- A job sharing a GPU with another user's process can run out of memory; keep one job per free GPU.
