# Running on HF Jobs (A10G recommended)

End-to-end pipeline runner for HuggingFace Jobs. A10G (Ampere, compute capability 8.6) avoids all the T4 limitations that blocked Colab: bf16, vLLM, Flash Attention 2, sane GRPO sampling.

## Cost ($1.05/hr A10G — $30 budget covers full run + lots of retries)

| Stage | Time | Cost |
|---|---|---|
| SFT generate + train | ~20 min | $0.35 |
| GRPO sanity (30 iters) | ~25 min | $0.45 |
| GRPO real (100 iters, staged) | ~60 min | $1.05 |
| Eval base + trained | ~10 min | $0.20 |
| **Full pipeline (`STAGE=all`)** | **~2 hr** | **~$2.05** |

## Prerequisites

```bash
pip install -U huggingface_hub
hf auth login   # use a write-scope token
```

## One-shot full pipeline

```bash
hf jobs run \
  --flavor a10g-large \
  --secrets HF_TOKEN=$(hf auth token) \
  --env HF_USER=YOUR_HF_USERNAME \
  --env GITHUB_REPO_URL=https://github.com/YOUR_GH_USERNAME/code-debugg-rl.git \
  --env STAGE=all \
  --env OUTER_ITERATIONS=100 \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  bash -lc "curl -sSL \$GITHUB_REPO_URL/raw/main/scripts/run_hf_job.sh | bash"
```

Replace:
- `YOUR_HF_USERNAME` → your HF username (e.g. `hemangtk`)
- `YOUR_GH_USERNAME` → your GitHub username if your fork is elsewhere

The script self-clones, installs deps, runs SFT + GRPO + eval, and uploads `artifacts/` to `YOUR_HF_USERNAME/qwen3-0.6b-arg-results`. The trained LoRA also pushes to `YOUR_HF_USERNAME/qwen3-0.6b-arg` (Hub model repo).

## Stage-by-stage (cheaper iteration)

If you want to iterate without paying for the whole pipeline each time, set `STAGE` to one of: `baseline`, `sft`, `sanity`, `grpo`, `eval`.

```bash
# Just measure the base model — $0.20
hf jobs run --flavor a10g-large \
  --secrets HF_TOKEN=$(hf auth token) \
  --env HF_USER=hemangtk \
  --env GITHUB_REPO_URL=https://github.com/hemangtk/code-debugg-rl.git \
  --env STAGE=baseline \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  bash -lc "curl -sSL \$GITHUB_REPO_URL/raw/main/scripts/run_hf_job.sh | bash"

# Run only SFT — $0.35
... --env STAGE=sft ...

# Run sanity GRPO (assumes SFT'd checkpoint already on Hub) — $0.45
... --env STAGE=sanity ...

# Real GRPO — $1.05
... --env STAGE=grpo ...
```

## Monitoring a running job

```bash
hf jobs ls                    # list active jobs
hf jobs logs JOB_ID --follow  # tail logs
hf jobs cancel JOB_ID         # kill if reward collapses
```

## Outputs

After a successful `STAGE=all`, you'll find:

- **`YOUR_HF_USERNAME/qwen3-0.6b-arg`** (model repo) — trained LoRA + tokenizer
- **`YOUR_HF_USERNAME/qwen3-0.6b-arg-results`** (model repo, used as artifact bucket):
  - `baseline.json` — pure base model metrics (acc, cave_rate, etc.)
  - `trained_results.json` — GRPO'd model metrics
  - `phase3_train.jsonl`, `phase3_episodes.jsonl` — sanity-train logs
  - `phase4_train.jsonl`, `phase4_episodes.jsonl` — real-train logs
  - `sft_data.jsonl` — generated SFT examples (for reproducibility)

## If GRPO sanity reward stays flat

Bump `--env GRPO_TEMPERATURE=1.5` (script reads it via env if set — TODO: thread it through if needed). Or fall back to base model:

```bash
... --env STAGE=grpo --env BASE_MODEL=Qwen/Qwen3-0.6B ...
```

## Adapting for A100 (full Phase 4: 300 iters)

```bash
hf jobs run --flavor a100-large \
  --env OUTER_ITERATIONS=300 \
  ...rest same...
```

A100 80GB at $4/hr × 2.5 hr = $10. Use only after sanity passes on A10G.
