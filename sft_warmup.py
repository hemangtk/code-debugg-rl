"""SFT format-warmup data generator and trainer.

The hackathon guide (§3) recommends a small SFT warmup before RL when
the base model isn't reliably emitting your action format. Phase 2's
smoke test surfaces this — if you see ``defaulted_to_read_code`` or
``no_action_detected`` warnings on a fresh Qwen, that's the signal to
run this.

We don't need hand-written demonstrations: our heuristic policy already
emits perfect ``ACTION:/TOOL:/ARGS:`` strings with the right shapes.
This script:

1. Runs the heuristic agent for N episodes against the env.
2. Captures every (prompt, completion) pair as an SFT example.
3. Optionally: trains a LoRA on the base model with TRL's SFTTrainer.

Usage::

    # Generate dataset only (no GPU needed)
    python sft_warmup.py --generate-only --n 200 --out sft_data.jsonl

    # Generate + train (GPU)
    python sft_warmup.py --n 200 --model Qwen/Qwen3-0.6B \\
        --out sft_data.jsonl --output-dir ckpts/qwen-format

After SFT, plug the resulting checkpoint into Phase 3 / Phase 4 as the
``--model`` argument.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

from server.environment import AdversarialReasoningEnv, MAX_TURNS
from server.rollout import build_prompt, format_observation


def _heuristic_completion(env: AdversarialReasoningEnv) -> str:
    """Return a perfectly-formatted action for the current env state.

    Mirrors the heuristic in train.py but always uses the oracle (no
    caving). The completions feed an SFT loop, so we want the model to
    learn the *format*, not the heuristic's wrong-line strategy.
    """
    s = env._scenario
    log = env._tool_calls
    used = {c["tool_name"] for c in log}
    if "read_code" not in used:
        return ('ACTION: tool_call\nTOOL: read_code\nARGS: {}\n'
                'REASONING: I should read the code first to see the function.')
    if "run_tests" not in used:
        return ('ACTION: tool_call\nTOOL: run_tests\nARGS: {}\n'
                'REASONING: Running the tests to see which fail.')
    if not any(c["tool_name"] == "run_code" for c in log):
        failing = s.grading.failing_inputs[0] if s.grading.failing_inputs else ""
        return (
            f'ACTION: tool_call\nTOOL: run_code\n'
            f'ARGS: {{"input_value": "{failing}"}}\n'
            f'REASONING: Tracing the failing input to see what the code returns.'
        )
    if not env._modified_lines:
        line = s.bug.bug_line
        new_code = s.bug.original_code
        args_json = json.dumps({"line_number": line, "new_code": new_code})
        return (
            f'ACTION: tool_call\nTOOL: apply_fix\nARGS: {args_json}\n'
            f'REASONING: After tracing, the actual bug is on line {line}, '
            f'not the line the user pointed to. Fixing it.'
        )
    return ('ACTION: submit\nREASONING: All tests should pass now — fix verified.')


def generate_sft_data(n_episodes: int, seed: int, out_path: str,
                      difficulty: str = "easy") -> int:
    random.seed(seed)
    env = AdversarialReasoningEnv(seed=seed)
    examples: List[Dict] = []

    for ep in range(n_episodes):
        obs = env.reset(difficulty=difficulty)
        history: List[Dict] = []
        for _ in range(MAX_TURNS):
            prompt = build_prompt(history, obs)
            completion = _heuristic_completion(env)
            examples.append({"prompt": prompt, "completion": completion})
            history.append({"obs_text": format_observation(obs),
                            "completion": completion})
            # Apply the action to advance the env.
            from server.action_parser import parse_action
            action, _ = parse_action(completion)
            result = env.step(action)
            if result.done:
                break
            obs = result.observation

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} SFT examples to {out_path}")
    return len(examples)


def train_sft(args) -> None:  # pragma: no cover - GPU-only
    try:
        from datasets import load_dataset  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError as e:
        raise SystemExit(f"SFT mode requires transformers + trl + datasets. {e}")

    print(f"[sft] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()
        # gradient_checkpointing requires use_cache=False; otherwise HF prints a warning and silently disables it.
        if hasattr(model, "config"):
            model.config.use_cache = False

    print(f"[sft] loading dataset from {args.out}...")
    ds = load_dataset("json", data_files=args.out, split="train")

    from server.rollout import prompt_to_messages

    def _format(example):
        # Re-template the placeholder-token prompt through the tokenizer's
        # actual chat format so SFT trains on the same shape the model
        # sees at inference. Without this, the SFT'd model learns to
        # respond to fake `<|user|>` tokens and produces garbage at eval.
        msgs = prompt_to_messages(example["prompt"]) + [
            {"role": "assistant", "content": example["completion"]},
        ]
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
        else:
            text = example["prompt"] + example["completion"]
        return {"text": text}

    ds = ds.map(_format)

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        dataloader_num_workers=0,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[sft] saved warmup checkpoint to {args.output_dir}")
    print(f"[sft] now use this path as --model in Phase 3 / Phase 4.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200,
                   help="Episodes to roll out (each yields up to 12 examples)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty", default="easy")
    p.add_argument("--out", default="sft_data.jsonl")
    p.add_argument("--generate-only", action="store_true",
                   help="Only build the dataset; skip training (no GPU needed).")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", default="ckpts/sft-warmup")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-device batch. T4 with 0.6B + grad-checkpoint fits 1.")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Effective batch = batch_size * grad_accum.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=1024,
                   help="Truncate to this many tokens. 1024 fits T4; 2048 needs A100.")
    args = p.parse_args()

    n = generate_sft_data(args.n, args.seed, args.out, args.difficulty)
    if n == 0:
        raise SystemExit("Generated no examples. Check env / heuristic.")

    if args.generate_only:
        print("[sft] --generate-only set; skipping training.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    train_sft(args)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
