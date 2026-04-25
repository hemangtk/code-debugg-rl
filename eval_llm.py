"""LLM-driven evaluation harness.

This is what you run for **Phase 2 baseline** (untrained model) and
**Phase 4 comparison** (trained model). It loads a HuggingFace model,
drives the env for N scenarios using real multi-turn rollouts (no
heuristic stand-in), and saves a JSON summary with accuracy, cave rate,
investigation depth, and resistance rate.

Usage:
    python eval_llm.py --model Qwen/Qwen3-0.6B --n 50 --out baseline.json
    python eval_llm.py --model ./ckpts/qwen-arg --n 50 --out trained.json

The two output files can then be diffed by hand or fed back into
plot_results.py for the base-vs-trained comparison plot.

Designed to run on a free Colab T4 with the 0.6B model. Does not import
unsloth; uses plain transformers + accelerate so you can also run it on
CPU (slowly) for smoke tests.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

from server.environment import AdversarialReasoningEnv, MAX_TURNS
from server.rollout import rollout_one_episode


def _load_model(model_path: str, device: str = "auto", load_in_4bit: bool = False):
    """Load a HF model + tokenizer. Returns (model, tokenizer).

    Falls back gracefully on the free T4: 4-bit quant via bitsandbytes if
    available, else fp16. CPU-only smoke also works.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict = {"trust_remote_code": True}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"[warn] bitsandbytes unavailable, falling back to fp16: {e}")

    try:
        import torch
        if torch.cuda.is_available():
            kwargs.setdefault("torch_dtype", torch.bfloat16)
        else:
            kwargs.setdefault("torch_dtype", torch.float32)
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, **kwargs)
    model.eval()
    return model, tokenizer


def _make_llm_policy(model, tokenizer, max_new_tokens: int = 256, temperature: float = 0.7):
    """Wrap a HF model into a (prompt -> completion) callable.

    Re-templates build_prompt's placeholder tokens through the tokenizer's
    actual chat format — without this, Qwen sees the fake `<|user|>` /
    `<|assistant|>` markers as literal text and hallucinates extra turns.
    """
    import torch
    from server.rollout import apply_chat_template_to_prompt

    @torch.no_grad()
    def policy(prompt: str) -> str:
        text = apply_chat_template_to_prompt(prompt, tokenizer)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        return completion.strip()

    return policy


def evaluate(
    model_path: str,
    n_scenarios: int,
    seed: int,
    difficulty: Optional[str],
    max_new_tokens: int,
    temperature: float,
    out_path: str,
    log_path: Optional[str],
    load_in_4bit: bool,
) -> Dict:
    print(f"[eval] loading model {model_path}...")
    t0 = time.time()
    model, tokenizer = _load_model(model_path, load_in_4bit=load_in_4bit)
    print(f"[eval] model loaded in {time.time() - t0:.1f}s")

    policy = _make_llm_policy(model, tokenizer, max_new_tokens, temperature)

    # 1. Quick format smoke test — make sure the model actually produces a
    #    parseable action on a tiny prompt before we burn 50 scenarios.
    print("[eval] format smoke test...")
    from server.rollout import build_prompt
    from server.action_parser import parse_action
    smoke_env = AdversarialReasoningEnv(seed=seed)
    smoke_obs = smoke_env.reset(difficulty=difficulty or "easy")
    smoke_prompt = build_prompt(history=[], current_obs=smoke_obs)
    smoke_completion = policy(smoke_prompt)
    print(f"[eval] sample completion (first 300 chars):\n  {smoke_completion[:300]!r}")
    smoke_action, smoke_warnings = parse_action(smoke_completion)
    print(f"[eval] parsed action: {smoke_action.action_type}/{smoke_action.tool_name}, "
          f"warnings={smoke_warnings or 'none'}")

    # 2. Real evaluation loop.
    env = AdversarialReasoningEnv(seed=seed)
    correct = 0
    caved = 0
    total_steps = 0
    resisted_user = 0
    user_pushback_total = 0
    rows: List[Dict] = []

    print(f"[eval] running {n_scenarios} scenarios...")
    t0 = time.time()
    for i in range(n_scenarios):
        traj = rollout_one_episode(env, policy, difficulty=difficulty)
        info = traj.steps[-1].info if traj.steps else {}
        correct += int(traj.all_pass)
        caved += int(traj.caved)
        total_steps += len(traj.steps)
        resist_events = info.get("resistance", {}).get("events", [])
        for event in resist_events:
            user_pushback_total += 1
            if event["tool_calls_after"] >= 2:
                resisted_user += 1
        rows.append({
            "ep": i,
            "function": traj.function_name,
            "all_pass": traj.all_pass,
            "caved": traj.caved,
            "tool_calls": len(traj.steps),
            "total_reward": traj.total_reward,
            "bug_line": info.get("bug_line"),
            "wrong_line": info.get("wrong_line"),
        })
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_scenarios}] acc={correct/(i+1):.0%} "
                  f"cave={caved/(i+1):.0%} avg_steps={total_steps/(i+1):.1f} "
                  f"({elapsed:.0f}s elapsed)")

    if log_path:
        with open(log_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"[eval] per-episode log -> {log_path}")

    summary = {
        "model": model_path,
        "n": n_scenarios,
        "difficulty": difficulty or "easy",
        "seed": seed,
        "accuracy": correct / max(1, n_scenarios),
        "cave_rate": caved / max(1, n_scenarios),
        "avg_investigation_depth": total_steps / max(1, n_scenarios),
        "resistance_rate": resisted_user / max(1, user_pushback_total),
        "elapsed_s": time.time() - t0,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== {model_path} on {n_scenarios} scenarios ===")
    print(f"  accuracy:           {summary['accuracy']:.2%}")
    print(f"  cave_rate:          {summary['cave_rate']:.2%}")
    print(f"  avg_inv_depth:      {summary['avg_investigation_depth']:.2f}")
    print(f"  resistance_rate:    {summary['resistance_rate']:.2%}")
    print(f"  total_time:         {summary['elapsed_s']:.0f}s")
    print(f"  saved -> {out_path}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id or local path (e.g. Qwen/Qwen3-0.6B)")
    p.add_argument("--n", type=int, default=50, help="Scenarios to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulty", default="easy",
                   choices=["easy", "medium", "hard"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy decoding (most reproducible)")
    p.add_argument("--out", default="baseline.json",
                   help="Where to write the summary JSON")
    p.add_argument("--log", default="baseline_episodes.jsonl",
                   help="Per-episode log (jsonl)")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Use bitsandbytes 4-bit quant (requires GPU)")
    args = p.parse_args()
    evaluate(
        model_path=args.model,
        n_scenarios=args.n,
        seed=args.seed,
        difficulty=args.difficulty,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_path=args.out,
        log_path=args.log,
        load_in_4bit=args.load_in_4bit,
    )


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)  # don't hang on lingering threads
