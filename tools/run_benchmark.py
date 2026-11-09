#!/usr/bin/env python3
"""One-command CMBench runner: prepare requests -> generate -> evaluate -> summarize.

Example (all tasks, seed 2, two methods, one GPU):

    python tools/run_benchmark.py --methods decoprune snapkv --seeds 2 --gpus 0 \
        --checkpoint models/lingbot-world-v2-14b-causal-fast \
        --benchmark-root data/CMBench --model-root models --output-root runs/cmbench

Generation is resumable: a case whose ``continuation.mp4`` and ``metadata.json``
already exist is skipped.  ``--smoke`` shortens context/output for a quick
functional check (the result is marked diagnostic and is not a benchmark number).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmbench_rebuild.dataset import load_benchmark, prepare_requests  # noqa: E402
from cmbench_rebuild.methods.registry import REGISTRY, resolve  # noqa: E402

HEAD_MAP = ROOT / "configs" / "head_maps" / "lingbot_heads_q0_sink4_recent4_thr08_cal15_v2.json"
EVALUATOR = ROOT / "third_party" / "cmbench_evaluator" / "eval_context_memory_relevance.py"
PAPER_PARAMS = {
    "decoprune": {"step_index": 2, "generation_step_index": 2, "threshold": 0.1},
    "decoprune_hs": {"step_index": 2, "generation_step_index": 2, "threshold": 0.1},
    "random": {"step_index": 2, "generation_step_index": 2, "threshold": 0.1},
    "contextfree": {"step_index": 2, "generation_step_index": 2, "threshold": 0.1},
}


def method_config(method: str, *, checkpoint: Path, requests: Path, world_size: int, overrides: dict, gpu_memory_gib: float = 80.0) -> dict:
    spec = resolve(method)
    config = {
        "protocol": "cmbench-116",
        "method": spec.id,
        "case_id": "unused",
        "seed": 2,
        "checkpoint": str(checkpoint.resolve()),
        "dataset_version": "Aoraku/CMBench",
        "request_file": str(requests.resolve()),
        "world_size": int(world_size),
        "gpu_memory_limit_gib": float(gpu_memory_gib),
        "rope_reindex": {"mode": "all_bands"},
        "method_params": {**PAPER_PARAMS.get(spec.id, {}), **overrides.get(spec.id, {})},
    }
    if spec.id in {"forcingkv", "decoprune_hs"}:
        config["head_map_file"] = str(HEAD_MAP)
    return config


def run_generation(args, method: str, config_path: Path, case_id: str, seed: int, output: Path) -> dict:
    if (output / "continuation.mp4").is_file() and (output / "metadata.json").is_file():
        return {"status": "skipped"}
    if output.exists():  # a previous attempt failed part-way; production refuses existing outputs
        output.rename(output.with_name(f"{output.name}.failed-{int(time.time())}"))
    output.parent.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
        "-m", "cmbench_rebuild.production", str(config_path),
        "--output", str(output), "--case-id", case_id, "--seed", str(seed),
    ]
    if args.smoke:
        command += ["--context-latent-limit", str(args.smoke_context_frames), "--output-latent-frames", str(args.smoke_output_frames)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    started = time.time()
    with output.with_name(f"{output.name}.generation.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status = "ok" if completed.returncode == 0 and (output / "continuation.mp4").is_file() else f"failed({completed.returncode})"
    return {"status": status, "seconds": time.time() - started}


def evaluate(args, method: str, seed: int, case_dirs: list[Path]) -> None:
    from cmbench_rebuild.evaluation.official import run_official_dino_batch

    eval_root = args.output_root / "eval" / method / f"seed{seed}"
    pending = [d for d in case_dirs if not (eval_root / "done" / f"{d.parent.name}.json").is_file()]
    if not pending:
        return
    model_root = args.model_root
    metadata = args.benchmark_root / "metadata.jsonl"
    for start in range(0, len(pending), args.eval_batch):
        batch = pending[start:start + args.eval_batch]
        results = run_official_dino_batch(
            case_dirs=batch, benchmark_root=args.benchmark_root, metadata=metadata, annotations=metadata,
            evaluator=EVALUATOR, python=Path(args.eval_python) if args.eval_python else Path(sys.executable),
            owl_model=model_root / "google" / "owlvit-base-patch32",
            sam_model=model_root / "facebook" / "sam2.1-hiera-large",
            dino_model=model_root / "facebook" / "dinov2-base",
            eval_root=eval_root / f"batch{start // args.eval_batch:04d}", device="cuda:0",
        )
        (eval_root / "done").mkdir(parents=True, exist_ok=True)
        for case_id, result in results.items():
            (eval_root / "done" / f"{case_id}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")


def _per_frame_topk(eval_case: dict, k: int) -> float | None:
    """Mean of the top-k per-frame DINO scores (robustness check for the max aggregation)."""
    summary = eval_case.get("summary", {})
    frames = summary.get("per_frame_scores") or summary.get("frame_scores")
    if not frames:
        return None
    values = sorted((float(v) for v in frames if v is not None and math.isfinite(float(v))), reverse=True)
    return sum(values[:k]) / max(1, len(values[:k])) if values else 0.0


def bootstrap_ci(values: list[float], *, resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def summarize(args, methods: list[str], seeds: list[int], cases: list[str]) -> None:
    rows, table = [], []
    for method in methods:
        for seed in seeds:
            for case_id in cases:
                run = args.output_root / "runs" / method / case_id / f"seed{seed}"
                done = args.output_root / "eval" / method / f"seed{seed}" / "done" / f"{case_id}.json"
                meta = json.loads((run / "metadata.json").read_text()) if (run / "metadata.json").is_file() else {}
                result = json.loads(done.read_text()) if done.is_file() else {}
                seqpr = (meta.get("seq_pr_metrics") or {})
                rows.append({
                    "method": method, "seed": seed, "case_id": case_id,
                    "dino": result.get("dino_score"),
                    "dino_top5": _per_frame_topk(result, 5) if result else None,
                    "seq_pr": seqpr.get("sequence_prune_ratio", seqpr.get("seq_pr")),
                    "fps": (meta.get("resources") or {}).get("decoded_frame_fps_nominal"),
                })
    out = args.output_root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Method | N | DINO | 95% CI | DINO top-5 | seqPR |", "|---|---|---|---|---|---|"]
    for method in methods:
        scores = [r["dino"] for r in rows if r["method"] == method and r["dino"] is not None]
        top5 = [r["dino_top5"] for r in rows if r["method"] == method and r["dino_top5"] is not None]
        prs = [r["seq_pr"] for r in rows if r["method"] == method and r["seq_pr"] is not None]
        mean = sum(scores) / len(scores) if scores else float("nan")
        lo, hi = bootstrap_ci(scores)
        lines.append(f"| {method} | {len(scores)} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | "
                     f"{(sum(top5) / len(top5)) if top5 else float('nan'):.4f} | {(sum(prs) / len(prs)) if prs else float('nan'):.4f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", required=True, help=f"any of: all, {', '.join(REGISTRY)}")
    parser.add_argument("--cases", nargs="*", help="task ids (default: all 116)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2])
    parser.add_argument("--gpus", default="0", help="comma-separated CUDA ids for one generation job")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True, help="holds google/owlvit-base-patch32, facebook/sam2.1-hiera-large, facebook/dinov2-base")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-params", type=json.loads, default={}, help='JSON overrides, e.g. {"snapkv": {"keep_ratio": 0.15}}')
    parser.add_argument("--output-latent-frames", type=int, default=16)
    parser.add_argument("--gpu-memory-gib", type=float, default=80.0, help="per-GPU budget for weights+KV; e.g. 250 on an idle B300, less when sharing a card")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-python", help="interpreter of the evaluator environment (transformers==5.12.1, see tools/setup_env.sh)")
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-context-frames", type=int, default=24)
    parser.add_argument("--smoke-output-frames", type=int, default=8)
    args = parser.parse_args()
    methods = list(REGISTRY) if args.methods == ["all"] else [resolve(m).id for m in args.methods]
    args.output_root.mkdir(parents=True, exist_ok=True)
    requests = args.output_root / "inputs" / "requests.jsonl"
    if not requests.is_file():
        prepare_requests(args.benchmark_root, requests, args.seeds, output_latent_frames=args.output_latent_frames)
    cases = args.cases or [row["task_id"] for row in load_benchmark(args.benchmark_root)]
    log = (args.output_root / "progress.jsonl").open("a")
    for method in methods:
        config_path = args.output_root / "configs" / f"{method}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(method_config(method, checkpoint=args.checkpoint, requests=requests,
                                                        world_size=len(args.gpus.split(",")), overrides=args.method_params,
                                                        gpu_memory_gib=args.gpu_memory_gib), indent=2) + "\n")
        for seed in args.seeds:
            done_dirs = []
            for case_id in cases:
                output = args.output_root / "runs" / method / case_id / f"seed{seed}"
                status = run_generation(args, method, config_path, case_id, seed, output)
                log.write(json.dumps({"method": method, "seed": seed, "case_id": case_id, **status}) + "\n")
                log.flush()
                print(f"[{method} seed{seed}] {case_id}: {status}", flush=True)
                if (output / "continuation.mp4").is_file():
                    done_dirs.append(output)
            if not args.skip_eval and done_dirs:
                evaluate(args, method, seed, done_dirs)
    if not args.skip_eval:
        summarize(args, methods, args.seeds, cases)


if __name__ == "__main__":
    main()
