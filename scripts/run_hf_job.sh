#!/usr/bin/env bash
# End-to-end pipeline for HF Jobs (A10G or A100). Runs:
#   1. SFT warmup (generate + train) — ~20 min on A10G
#   2. GRPO sanity check (30 iters, easy)            — ~30 min on A10G
#   3. GRPO real training (100 iters, staged)        — ~60 min on A10G
#   4. Eval base + trained, push to Hub
#
# Required env vars (set via `hf jobs run --secrets`):
#   HF_TOKEN          : write-scope token
#   HF_USER           : your HF username (e.g. "hemangtk")
#   GITHUB_REPO_URL   : public git URL to clone (e.g. https://github.com/hemangtk/code-debugg-rl.git)
#
# Optional:
#   BASE_MODEL        : base model id (default Qwen/Qwen3-0.6B)
#   STAGE             : "sft", "sanity", "grpo", "eval", or "all" (default all)
#   OUTER_ITERATIONS  : GRPO iterations (default 100)
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
STAGE="${STAGE:-all}"
OUTER_ITERATIONS="${OUTER_ITERATIONS:-100}"
HUB_REPO="${HF_USER}/qwen3-0.6b-arg"
RESULTS_REPO="${HF_USER}/qwen3-0.6b-arg-results"

echo "=== HF Jobs run starting ==="
echo "  base_model=$BASE_MODEL stage=$STAGE outer=$OUTER_ITERATIONS"
echo "  hub_repo=$HUB_REPO results_repo=$RESULTS_REPO"
nvidia-smi || true

cd /tmp
git clone "$GITHUB_REPO_URL" code-debugg-rl
cd code-debugg-rl

# Triton JIT-compiles CUDA kernels at runtime (used by Unsloth's fused
# RMSNorm). The pytorch -runtime image has no C compiler, so the first
# kernel call dies with "Failed to find C compiler". Install build-essential
# before pip so any compile-from-source wheels also work.
echo "[setup] installing build-essential for Triton JIT"
apt-get update -qq && apt-get install -y --no-install-recommends \
  build-essential gcc g++ \
  > /tmp/apt.log 2>&1 || { echo "[fatal] apt install failed"; cat /tmp/apt.log; exit 1; }
export CC=gcc CXX=g++

pip install --no-cache-dir -r requirements-train.txt

# Uninstall torchvision entirely. We don't use image features, but
# transformers/image_utils does `import torchvision` inside a try/except
# that only catches ImportError — a *broken* torchvision (version mismatch
# with torch from transitive deps like bitsandbytes) raises RuntimeError
# ("operator torchvision::nms does not exist") which isn't caught and
# crashes any transformers import. Cleanly removing it makes the
# try/except behave correctly.
pip uninstall -y torchvision || true

# Smoke test the stack before we burn GPU time.
python -c "import torch, transformers; \
  print(f'torch={torch.__version__} tx={transformers.__version__}')" \
  || { echo "[fatal] import smoke test failed"; exit 1; }

mkdir -p artifacts ckpts

# Pull the SFT checkpoint from the results repo if we don't already have
# it locally. Lets STAGE=grpo / sanity / eval skip the (re)training of
# SFT when a previous run already produced one.
ensure_sft_warmup() {
  if [ -f ckpts/sft-warmup/model.safetensors ]; then
    echo "[setup] using existing ckpts/sft-warmup"
    return
  fi
  echo "[setup] downloading sft-warmup from $RESULTS_REPO"
  RESULTS_REPO="$RESULTS_REPO" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["RESULTS_REPO"],
    repo_type="model",
    allow_patterns=["sft-warmup/*"],
    local_dir="ckpts",
)
PY
  if [ ! -f ckpts/sft-warmup/model.safetensors ]; then
    echo "[fatal] failed to download sft-warmup from $RESULTS_REPO" >&2
    exit 1
  fi
}

run_sft() {
  echo "=== [1/4] SFT warmup ==="
  python sft_warmup.py --generate-only --n 400 --out sft_data.jsonl --seed 0
  python sft_warmup.py \
    --model "$BASE_MODEL" --out sft_data.jsonl \
    --output-dir ckpts/sft-warmup \
    --epochs 2 --batch-size 4 --grad-accum 2 --lr 2e-5 --max-length 2048
  cp sft_data.jsonl artifacts/
}

run_eval_base() {
  echo "=== eval base ==="
  python eval_llm.py --model "$BASE_MODEL" --n 50 --seed 42 --difficulty easy \
    --temperature 0.0 --max-new-tokens 512 \
    --out artifacts/baseline.json --log artifacts/baseline_episodes.jsonl
}

run_sanity() {
  ensure_sft_warmup
  echo "=== [2/4] GRPO sanity (30 iters, easy) ==="
  python train.py \
    --model ./ckpts/sft-warmup \
    --outer-iterations 30 --rollouts-per-iter 4 --num-generations 4 \
    --max-new-tokens 256 --max-seq-length 2048 \
    --lr 1e-5 --grad-accum 4 --gamma 0.95 \
    --grpo-temperature 1.2 \
    --curriculum easy \
    --output-dir ckpts/phase3-sanity \
    --output-log artifacts/phase3_train.jsonl \
    --episodes-log artifacts/phase3_episodes.jsonl --seed 0
}

run_grpo() {
  ensure_sft_warmup
  echo "=== [3/4] GRPO real (${OUTER_ITERATIONS} iters, staged) ==="
  python train.py \
    --model ./ckpts/sft-warmup \
    --outer-iterations "$OUTER_ITERATIONS" --rollouts-per-iter 6 --num-generations 6 \
    --max-new-tokens 256 --max-seq-length 2048 \
    --lr 1e-5 --grad-accum 4 --gamma 0.95 \
    --grpo-temperature 1.2 \
    --curriculum phase4 \
    --save-every 25 \
    --hub-repo "$HUB_REPO" \
    --output-dir ckpts/phase4-real \
    --output-log artifacts/phase4_train.jsonl \
    --episodes-log artifacts/phase4_episodes.jsonl --seed 0
}

run_eval_trained() {
  echo "=== [4/4] eval trained ==="
  python eval_llm.py --model ./ckpts/phase4-real --n 50 --seed 42 --difficulty easy \
    --temperature 0.0 --max-new-tokens 512 \
    --out artifacts/trained_results.json --log artifacts/trained_episodes.jsonl
}

push_artifacts() {
  echo "=== uploading artifacts to $RESULTS_REPO ==="
  python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo = f"{os.environ['HF_USER']}/qwen3-0.6b-arg-results"
create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path="artifacts", repo_id=repo, repo_type="model")
print(f"Uploaded artifacts to https://huggingface.co/{repo}")
PY
}

push_artifacts_safe() {
  # Best-effort artifact upload — don't fail the script if it errors.
  # Also rsync any in-progress checkpoints into artifacts/ so they survive.
  if [ -d ckpts/sft-warmup ]; then
    cp -r ckpts/sft-warmup artifacts/ 2>/dev/null || true
  fi
  if [ -d ckpts/phase4-real ]; then
    cp -r ckpts/phase4-real artifacts/ 2>/dev/null || true
  fi
  push_artifacts || echo "[warn] artifact upload failed (continuing)"
}

# Always attempt to save artifacts on exit, even on failure — protects
# expensive SFT checkpoint if a later stage crashes.
trap push_artifacts_safe EXIT

case "$STAGE" in
  sft)        run_sft ;;
  sanity)     run_sanity ;;
  grpo)       run_grpo; run_eval_trained ;;
  eval)       run_eval_base; run_eval_trained ;;
  baseline)   run_eval_base ;;
  all)
    run_eval_base
    run_sft
    run_sanity
    run_grpo
    run_eval_trained
    ;;
  *)          echo "unknown STAGE: $STAGE" >&2; exit 1 ;;
esac

echo "=== done ==="
