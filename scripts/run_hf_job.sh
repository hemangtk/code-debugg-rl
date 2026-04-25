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

pip install --no-cache-dir -r requirements-train.txt

mkdir -p artifacts ckpts

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

case "$STAGE" in
  sft)        run_sft ;;
  sanity)     run_sanity ;;
  grpo)       run_grpo ;;
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

push_artifacts
echo "=== done ==="
