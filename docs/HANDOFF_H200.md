# Handoff: 7 extra LingBot methods on CMBench (H200)

**Task:** run seed 2 on all 116 CMBench tasks for these 7 methods:
`random`, `contextfree`, `snapkv`, `focused_forcing`, `worldkv`, `future_forcing`, `densitykv`.
Each run produces a `continuation.mp4` per task, the official OWL/SAM/DINO score,
seqPR and FPS, and a summary table (mean DINO with a 95% bootstrap CI, top-5-frame
DINO, seqPR, FPS).

The Table 1 methods (FullKV, Streaming, DummyForcing, ForcingKV, TempDiff, DeCoPrune,
DeCoPrune-HS) are run separately and are **not** part of this handoff.

---

## 1. Where everything is

| What | Where |
|---|---|
| Code | https://github.com/Aoraku/CMBench (`main`) |
| Checkpoint | HF `robbyant/lingbot-world-v2-14b-causal-fast` (81 GB) |
| Benchmark | HF dataset `Aoraku/CMBench` (58 videos, 116 tasks, `metadata.jsonl`) |
| Evaluator models | HF `google/owlvit-base-patch32`, `facebook/sam2.1-hiera-large`, `facebook/dinov2-base` |
| Evaluator code | `third_party/cmbench_evaluator/eval_context_memory_relevance.py` (bundled) |
| Method defaults (budgets) | `src/cmbench_rebuild/methods/<method>.py`, `SPEC` defaults; already calibrated |
| What each new method does, and where it deviates from its paper | `docs/rebuttal_methods.md` |
| One-command runner | `tools/run_benchmark.py` |
| Offline PR check for a budget change | `tools/simulate_pr.py` |

## 2. Setup (once)

```bash
git clone https://github.com/Aoraku/CMBench && cd CMBench
bash tools/setup_env.sh          # .venv (generation) + .venv-eval (evaluator, transformers==5.12.1)
bash tools/download_assets.sh    # models/ and data/CMBench; also copies config.json into transformers/
```

- `ffmpeg` and `ffprobe` must be on `PATH` (a static build works if there is no system package).
- No internet on the GPU nodes? Download on a machine that has it and copy `models/` and `data/` over.
- H200 runs flash-attn 2 normally. On Blackwell GPUs (B200/B300) the code automatically falls back to PyTorch SDPA.

## 3. The 7 methods

| Method | `--methods` id | Budget (default, already calibrated) | Expected seqPR |
|---|---|---|---|
| Random (matched budget) | `random` | DeCoPrune's per-chunk keep count, random tokens | ≈0.853 (paper Table 3) |
| ContextFree | `contextfree` | DeCoPrune's per-chunk keep count, ranked by the no-cache discrepancy | ≈ DeCoPrune (0.863 / 0.865 on our smoke tasks) |
| SnapKV | `snapkv` | `keep_ratio=0.10` | 0.857 |
| FocusedForcing | `focused_forcing` | `frame_budget=36` frames | 0.856 |
| WorldKV | `worldkv` | `retrieved_chunks=14`, camera-pose retrieval | 0.854 |
| FutureForcing | `future_forcing` | `keep_ratio=0.10` | 0.857 |
| DensityKV | `densitykv` | `max_entries=37500` per layer, `growth_limit=8.0` | 0.856 |

All defaults are calibrated to DeCoPrune's pruning ratio (85.43% in the paper), so no
`--method-params` are needed. RoPE re-indexing (`all_bands`) is applied to every
method, as in Table 1. DeCoPrune's probe settings (step 2, γ=0.10) are set by the
runner for `random` and `contextfree`.

## 4. Commands (H200)

Run one job per GPU. Each job can take a list of methods, which run one after another.
Multi-GPU sequence parallel (`--gpus 0,1`) also works for all 7 methods (2-GPU smoke
passed), but one GPU per job is the simplest. On a 141 GB H200 set
`--gpu-memory-gib 130`; KV that does not fit is kept in host memory automatically.

Jobs that share an `--output-root` must not start in the same second, because the
first one writes `inputs/requests.jsonl`. Start them about 30 s apart.

```bash
R=runs/rebuttal_seed2
COMMON="--seeds 2 --gpu-memory-gib 130 --checkpoint models/lingbot-world-v2-14b-causal-fast \
  --benchmark-root data/CMBench --model-root models --output-root $R --eval-python .venv-eval/bin/python"

# 7 GPUs, one method each (the slowest methods first)
.venv/bin/python tools/run_benchmark.py --methods focused_forcing --gpus 0 $COMMON &
sleep 30
.venv/bin/python tools/run_benchmark.py --methods densitykv       --gpus 1 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods worldkv         --gpus 2 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods contextfree     --gpus 3 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods future_forcing  --gpus 4 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods snapkv          --gpus 5 $COMMON &
.venv/bin/python tools/run_benchmark.py --methods random          --gpus 6 $COMMON &
```

**Splitting a slow method over several GPUs:** give each job a disjoint `--cases ...`
list (task ids are the `task_id` field of `data/CMBench/metadata.jsonl`), with the
same `--output-root`. For example, split `focused_forcing` into four quarters of the
116 ids on four GPUs.

**Resuming:** re-running the same command skips tasks whose `continuation.mp4` and
`metadata.json` already exist. A task that failed part-way is moved to
`<task>/seed2.failed-<time>` and regenerated.

**Time per task** (full 60 s context, one GPU; measured on our smoke runs, on GPUs
shared with other jobs, so treat these as rough):

| Method | per task | 116 tasks on one GPU |
|---|---|---|
| Random | ≈ DeCoPrune (5.5 min on B300, ~8–10 min on H200) | ~15–20 h |
| SnapKV, FutureForcing | 12–15 min | ~25–30 h |
| ContextFree | 14–15 min (one extra no-cache probe per chunk) | ~28 h |
| WorldKV | 22–39 min (retrieval gathers stored chunks every call) | ~45–75 h |
| DensityKV | ~50 min (Riesz density against the whole bank every chunk) | ~95 h |
| FocusedForcing | 30–49 min (keeps the full history; sparse attention) | ~60–95 h |

DensityKV, FocusedForcing and WorldKV should be split over several GPUs with `--cases`.

## 5. Outputs

```
$R/inputs/requests.jsonl                     frozen requests (116 tasks)
$R/configs/<method>.json                     exact config used per method
$R/runs/<method>/<task>/seed2/               continuation.mp4, metadata.json, seq_pr_metrics.json, result.json, events.rank0.jsonl
$R/runs/<method>/<task>/seed2.generation.log per-task log (look here first when a task fails)
$R/eval/<method>/seed2/done/<task>.json      official OWL/SAM/DINO result per task
$R/progress.jsonl                            one line per task attempt (status, seconds)
$R/summary/table.md, per_case.csv            final table: DINO, 95% CI, top-5-frame DINO, seqPR, FPS
```

The summary is rewritten at the end of every invocation. To rebuild it at any time,
re-run the same command: finished tasks are skipped, so it only re-summarizes.
Please send back `$R/summary/` and `$R/progress.jsonl` (and, if possible,
`$R/runs/*/*/seed2/metadata.json` and `seq_pr_metrics.json`).

## 6. Sanity checks after the first few tasks

1. `seqPR` in `$R/runs/<method>/<task>/seed2/seq_pr_metrics.json` is close to the expected value in §3. If DensityKV drifts far from 0.856, tell us before continuing.
2. `events.rank0.jsonl` contains `rebuttal_chunk_finalized` (SnapKV, FutureForcing, DensityKV, WorldKV) or `consistency_physical_compaction` (Random, ContextFree).
3. WorldKV: `result.json` → `generation.online_policy.rebuttal.visible_subset.retrieval.mode` is `camera` (pose retrieval) or `query` (fallback when the context camera could not be estimated). Both are expected.
4. Our smoke results on `syn_001_reappear_01` / `_02` (seed 2, full context), for comparison:

| Method | seqPR | DINO |
|---|---|---|
| ContextFree | 0.863 / 0.865 | 0.698 / 0.640 |
| SnapKV | 0.857 / 0.857 | 0.622 / 0.501 |
| FocusedForcing | 0.856 / 0.856 | 0.556 / 0.612 |
| WorldKV | 0.854 / 0.854 | 0.653 / 0.397 |
| FutureForcing | 0.857 / 0.857 | 0.707 / 0.612 |
| DensityKV | 0.856 / 0.856 | 0.821 / 0.800 |
| (DeCoPrune, reference) | 0.863 / 0.866 | 0.798 / 0.609 |

## 7. Known pitfalls (already fixed on `main`)

- The HF checkpoint lacks `transformers/config.json`: `tools/download_assets.sh` copies it.
- Long target labels used to crash the evaluator: OWL-ViT text is truncated to 16 tokens.
- Slow CPU T5 prompt encoding under torchrun (`OMP_NUM_THREADS=1`): the runner sets 16.
- A GPU shared with another user's job can run out of memory mid-task. Keep one job per free GPU; failed tasks are simply re-run on the next invocation.
- FPS/speedup is only comparable when every method (and FullKV) runs on the same, otherwise idle GPUs. If speed numbers are needed, measure them separately on dedicated GPUs.
