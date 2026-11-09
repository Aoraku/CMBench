#!/usr/bin/env bash
# Download the checkpoint, the benchmark and the evaluator models into ./models and ./data.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download robbyant/lingbot-world-v2-14b-causal-fast --local-dir models/lingbot-world-v2-14b-causal-fast
hf download Aoraku/CMBench --repo-type dataset --local-dir data/CMBench
hf download google/owlvit-base-patch32 --local-dir models/google/owlvit-base-patch32
hf download facebook/sam2.1-hiera-large --local-dir models/facebook/sam2.1-hiera-large
hf download facebook/dinov2-base --local-dir models/facebook/dinov2-base
# The Hugging Face release keeps config.json at the repo root; the causal-fast loader reads it from transformers/.
cp -n models/lingbot-world-v2-14b-causal-fast/config.json models/lingbot-world-v2-14b-causal-fast/transformers/config.json
