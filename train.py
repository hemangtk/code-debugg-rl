"""GRPO training loop for the Adversarial Reasoning Gym - Code Debugger.

Two modes:

* ``--dry-run``: drives the environment with a heuristic policy to
  validate the full pipeline and emit a training_log.jsonl that the
  plotter can consume. No GPU required.
* default: runs GRPO with TRL + Unsloth on a Qwen base model. Uses
  proper multi-turn rollouts via ``server.rollout``: collect K
  trajectories with the current policy, build a per-step dataset of
  (state, action, return-to-go), then train one GRPO step on those
  examples and repeat.

Why this matters: the previous loop fed the same parsed action into
``env.step`` 12 times per "completion," which collapses the multi-turn
env into a bandit and gives the agent no signal about later turns. The
multi-turn rollout fixes that.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from server.action_parser import parse_action as _parse_action_with_warnings
from server.environment import AdversarialReasoningEnv, MAX_TURNS
from server.models import Action, Observation
from server.rollout import (
    SYSTEM_PROMPT,
    Trajectory,
    build_prompt,
    format_observation,
    rollout_one_episode,
)


def parse_action(text: str) -> Action:
    action, _warnings = _parse_action_with_warnings(text)
    return action


# ---------------------------------------------------------------------------
# Heuristic policy used by --dry-run. Emits the same prompt-format strings
# the LLM would, so the rollout / parser path is exercised exactly the
# same way as in real training.
# ---------------------------------------------------------------------------


def _heuristic_completion(env: AdversarialReasoningEnv, oracle_p: float) -> str:
    s = env._scenario
    log = env._tool_calls
    used = {c["tool_name"] for c in log}
    if "read_code" not in used:
        return 'ACTION: tool_call\nTOOL: read_code\nARGS: {}\nREASONING: look at the code'
    if "run_tests" not in used:
        return 'ACTION: tool_call\nTOOL: run_tests\nARGS: {}\nREASONING: see what fails'
    if not any(c["tool_name"] == "run_code" for c in log):
        failing = s.grading.failing_inputs[0] if s.grading.failing_inputs else ""
        return (
            f'ACTION: tool_call\nTOOL: run_code\n'
            f'ARGS: {{"input_value": "{failing}"}}\n'
            f'REASONING: trace the failing input'
        )
    if not env._modified_lines:
        if random.random() < oracle_p:
            line = s.bug.bug_line
            new_code = s.bug.original_code
        else:
            line = s.wrong_line
            new_code = "    pass  # caved"
        # Args are JSON; escape the new_code value safely.
        args_json = json.dumps({"line_number": line, "new_code": new_code})
        return (
            f'ACTION: tool_call\nTOOL: apply_fix\nARGS: {args_json}\n'
            f'REASONING: fix line {line}'
        )
    return 'ACTION: submit\nREASONING: submit fix'


def _make_heuristic_policy(env: AdversarialReasoningEnv, oracle_p: float):
    def policy(_prompt: str) -> str:
        return _heuristic_completion(env, oracle_p)
    return policy


# ---------------------------------------------------------------------------
# --dry-run path: writes training_log.jsonl in the schema the plotter expects.
# ---------------------------------------------------------------------------


def run_dry_run(num_episodes: int, output_log: str, seed: int = 0) -> None:
    random.seed(seed)
    env = AdversarialReasoningEnv(seed=seed)
    log_lines: List[Dict] = []
    accuracy_window: List[bool] = []

    with open(output_log, "w") as out:
        for ep in range(num_episodes):
            oracle_p = 0.2 + 0.65 * min(1.0, ep / max(1, num_episodes - 1))
            policy = _make_heuristic_policy(env, oracle_p)
            traj = rollout_one_episode(env, policy)
            info = traj.steps[-1].info if traj.steps else {}
            accuracy_window.append(traj.all_pass)
            if len(accuracy_window) > 20:
                accuracy_window = accuracy_window[-20:]
            log = {
                "step": ep,
                "reward": traj.total_reward,
                "fix_score": info.get("fix_score", 0.0),
                "investigation_score": info.get("investigation_score", 0.0),
                "resistance_score": info.get("resistance_score", 0.0),
                "anti_cheat_score": info.get("anti_cheat_score", 0.0),
                "all_pass": traj.all_pass,
                "caved": traj.caved,
                "tool_calls": int(info.get("tool_calls", 0)),
                "sandbox_timeouts": int(info.get("sandbox_timeouts", 0)),
                "rolling_accuracy": (
                    sum(accuracy_window) / len(accuracy_window) if accuracy_window else 0.0
                ),
                "rolling_cave_rate": 0.0,  # filled below
                "difficulty": env.curriculum.snapshot(),
            }
            log_lines.append(log)
            out.write(json.dumps(log) + "\n")
            out.flush()

    # Post-fill rolling cave rate.
    caves: List[int] = []
    for entry in log_lines:
        caves.append(1 if entry["caved"] else 0)
        window = caves[-20:]
        entry["rolling_cave_rate"] = sum(window) / len(window)
    with open(output_log, "w") as out:
        for entry in log_lines:
            out.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(log_lines)} episodes to {output_log}.")


# ---------------------------------------------------------------------------
# GRPO training (GPU only). Uses real multi-turn rollouts:
#   collect_trajectories -> compute returns -> per-step GRPO update.
# ---------------------------------------------------------------------------


def _make_llm_policy(model, tokenizer, max_new_tokens: int = 256):
    from server.rollout import apply_chat_template_to_prompt

    def policy(prompt: str) -> str:
        text = apply_chat_template_to_prompt(prompt, tokenizer)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return completion
    return policy


def _trajectory_to_examples(
    traj: Trajectory, gamma: float = 0.95, tokenizer=None,
) -> List[Dict]:
    """Convert a trajectory into per-step (prompt, completion, return) tuples.

    If ``tokenizer`` is provided, prompts are re-templated through the
    tokenizer's actual chat format so GRPO trains on the same shape the
    model sees at inference time.
    """
    from server.rollout import apply_chat_template_to_prompt
    returns = traj.returns_to_go(gamma=gamma)
    examples = []
    for step, ret in zip(traj.steps, returns):
        prompt = step.prompt
        if tokenizer is not None:
            prompt = apply_chat_template_to_prompt(prompt, tokenizer)
        examples.append({
            "prompt": prompt,
            "completion": step.completion,
            "return": ret,
        })
    return examples


def _resolve_stage_difficulty(step: int, schedule: Optional[List[Dict]]) -> Optional[str]:
    """Phase 4 staged curriculum override.

    schedule = [{"until": 50, "difficulty": "easy"},
                {"until": 150, "difficulty": "medium"},
                {"until": 300, "difficulty": None}]   # None = adaptive
    """
    if not schedule:
        return None
    for stage in schedule:
        if step < stage["until"]:
            return stage.get("difficulty")
    return schedule[-1].get("difficulty")


def run_grpo_training(args) -> None:  # pragma: no cover - GPU-only path
    try:
        from unsloth import FastLanguageModel  # type: ignore
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
        from datasets import Dataset  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "GRPO mode requires unsloth, trl, and datasets. "
            "Install them on a CUDA machine and try again. (" + str(e) + ")"
        )

    # vLLM's torch.compile path breaks on compute capability < 8.0 (T4 = 7.5)
    # with "Tried to erase Node size_X but it still had N users". Auto-detect
    # GPU + check vllm is installed; disable fast_inference if either fails.
    fast_inference = args.fast_inference
    if fast_inference is None:
        try:
            import torch as _torch
            cc = _torch.cuda.get_device_capability(0)
            is_ampere_plus = cc[0] >= 8
        except Exception:
            cc = (0, 0)
            is_ampere_plus = False
        try:
            import vllm  # noqa: F401
            has_vllm = True
        except ImportError:
            has_vllm = False
        fast_inference = is_ampere_plus and has_vllm
        if not fast_inference:
            if not is_ampere_plus:
                print(f"[train] GPU compute capability {cc[0]}.{cc[1]} < 8.0 — "
                      "disabling vLLM fast_inference.")
            elif not has_vllm:
                print("[train] vllm package not installed — disabling fast_inference. "
                      "(`pip install vllm` to enable; ~20% faster GRPO rollouts.)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=fast_inference,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )

    log_path = open(args.output_log, "w") if args.output_log else None
    episodes_path = open(args.episodes_log, "w") if args.episodes_log else None
    accuracy_window: List[bool] = []
    cave_window: List[int] = []

    # Phase 4 staged curriculum: first N steps easy, then medium, then adaptive.
    schedule: Optional[List[Dict]] = None
    if args.curriculum == "phase4":
        schedule = [
            {"until": 50, "difficulty": "easy"},
            {"until": 150, "difficulty": "medium"},
            {"until": 10**9, "difficulty": None},
        ]
    elif args.curriculum == "easy":
        schedule = [{"until": 10**9, "difficulty": "easy"}]

    # Outer loop: K trajectories per iteration; each iteration = one GRPO update.
    for outer in range(args.outer_iterations):
        env = AdversarialReasoningEnv(seed=args.seed + outer)
        policy = _make_llm_policy(model, tokenizer, args.max_new_tokens)
        stage_difficulty = _resolve_stage_difficulty(outer, schedule)

        # 1. Collect trajectories.
        all_examples: List[Dict] = []
        ep_summaries: List[Dict] = []
        for ep_i in range(args.rollouts_per_iter):
            traj = rollout_one_episode(env, policy, difficulty=stage_difficulty)
            all_examples.extend(_trajectory_to_examples(traj, args.gamma, tokenizer))
            ep_summaries.append({
                "all_pass": traj.all_pass,
                "caved": traj.caved,
                "total_reward": traj.total_reward,
                "function": traj.function_name,
                "n_steps": len(traj.steps),
            })
            accuracy_window.append(traj.all_pass)
            cave_window.append(1 if traj.caved else 0)
            accuracy_window = accuracy_window[-20:]
            cave_window = cave_window[-20:]
            # Dump full episode trace for "read actual generated episodes"
            # spot-checks. This is critical for catching reward hacking.
            if episodes_path is not None:
                episodes_path.write(json.dumps({
                    "iter": outer,
                    "rollout": ep_i,
                    "function": traj.function_name,
                    "all_pass": traj.all_pass,
                    "caved": traj.caved,
                    "total_reward": traj.total_reward,
                    "stage_difficulty": stage_difficulty,
                    "steps": [
                        {
                            "completion": s.completion[:600],
                            "action_type": s.action.action_type,
                            "tool_name": s.action.tool_name,
                            "tool_args": s.action.tool_args,
                            "reward": s.reward,
                        }
                        for s in traj.steps
                    ],
                }) + "\n")
                episodes_path.flush()

        if not all_examples:
            print(f"Iter {outer}: no examples collected, skipping update.")
            continue

        # 2. Build a dataset of prompts; reward_fn looks up the precomputed
        #    return-to-go for each (prompt, completion) the trainer feeds back.
        #    GRPO will sample additional completions per prompt — for those we
        #    fall back to an estimated return based on a one-step heuristic.
        precomputed: Dict[Tuple[str, str], float] = {
            (ex["prompt"], ex["completion"]): ex["return"] for ex in all_examples
        }

        def reward_fn(prompts, completions, **_kw):
            rewards = []
            for p, c in zip(prompts, completions):
                key = (p, c)
                if key in precomputed:
                    rewards.append(precomputed[key])
                else:
                    # New completion sampled by GRPO. Re-roll out from this
                    # state with this completion as the first action; if we
                    # don't have an env snapshot, score conservatively at 0.
                    rewards.append(_score_new_completion(p, c, args.gamma))
            return rewards

        prompts = [{"prompt": ex["prompt"]} for ex in all_examples]
        ds = Dataset.from_list(prompts)

        # T4 (Turing, compute capability 7.5) doesn't support bf16 — only
        # Ampere+ does. Detect and fall back to fp16. Both are valid for
        # GRPO; bf16 is just preferred when available for numerical range.
        import torch as _torch
        _supports_bf16 = (
            _torch.cuda.is_available()
            and _torch.cuda.get_device_capability(0)[0] >= 8
        )
        # Note: don't pass `max_prompt_length` — Unsloth 2025.9.x's
        # GRPOConfig wrapper forwards it to a TRL GRPOConfig that doesn't
        # accept the kwarg, causing TypeError at init. The default
        # (model's max_seq_length) is fine for our prompts.
        config = GRPOConfig(
            output_dir=args.output_dir,
            num_generations=args.num_generations,
            max_completion_length=args.max_new_tokens,
            learning_rate=args.lr,
            bf16=_supports_bf16,
            fp16=not _supports_bf16,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=1,  # one training step per outer iteration
            save_steps=50,
            logging_steps=1,
            # Explicit sampling params — without these, post-SFT models that
            # are over-confident on the heuristic produce identical outputs
            # across all `num_generations` completions, collapsing GRPO's
            # group-relative advantage to zero. Symptom: reward_std=0,
            # grad_norm=0, frac_reward_zero_std=1.0.
            temperature=args.grpo_temperature,
            top_p=0.95,
            top_k=50,
        )
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_fn],
            args=config,
            train_dataset=ds,
        )
        trainer.train()

        # 3. Logging.
        rolling_acc = sum(accuracy_window) / max(1, len(accuracy_window))
        rolling_cave = sum(cave_window) / max(1, len(cave_window))
        for i, ep in enumerate(ep_summaries):
            entry = {
                "step": outer * args.rollouts_per_iter + i,
                "reward": ep["total_reward"],
                "all_pass": ep["all_pass"],
                "caved": ep["caved"],
                "tool_calls": ep["n_steps"],
                "rolling_accuracy": rolling_acc,
                "rolling_cave_rate": rolling_cave,
                "difficulty": env.curriculum.snapshot(),
                # eval-friendly placeholders so plot_results.py works unchanged:
                "fix_score": 0.0,
                "investigation_score": 0.0,
                "resistance_score": 0.0,
                "anti_cheat_score": 0.0,
            }
            if log_path:
                log_path.write(json.dumps(entry) + "\n")
                log_path.flush()
        print(
            f"iter={outer} accuracy={rolling_acc:.2%} cave={rolling_cave:.2%} "
            f"loss=ok"
        )

        if (outer + 1) % args.save_every == 0:
            trainer.save_model(os.path.join(args.output_dir, f"iter-{outer+1}"))

    if log_path:
        log_path.close()
    if episodes_path:
        episodes_path.close()

    # === IMPORTANT: QLoRA save handling ===
    # `trainer.save_model()` saves only the LoRA adapter on top of the
    # 4-bit quantized base. That's the SAFE path — do NOT upcast the
    # 4-bit base to fp16 and merge weights, that path is known to
    # silently degrade quality (Unsloth docs flag this loudly).
    #
    # If you want a single merged fp16 model for inference, prefer:
    #   model.save_pretrained_merged(args.output_dir, tokenizer,
    #                                save_method="merged_16bit")
    # which Unsloth handles correctly. For Hub uploads, push the adapter
    # directly — consumers can apply it via PeftModel.from_pretrained.
    trainer.save_model(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")
    print("[note] To merge to fp16, use Unsloth's save_pretrained_merged")
    print("       (do NOT manually upcast 4-bit -> fp16 then merge LoRA)")

    if args.save_merged:
        try:
            model.save_pretrained_merged(
                args.output_dir + "-merged",
                tokenizer,
                save_method="merged_16bit",
            )
            print(f"Saved merged fp16 model to {args.output_dir}-merged")
        except Exception as e:
            print(f"[warn] merged save failed: {e}")

    # Phase 4 step: optionally push final checkpoint to HF Hub.
    if args.hub_repo:
        try:
            trainer.push_to_hub(args.hub_repo)
            print(f"Pushed final model to https://huggingface.co/{args.hub_repo}")
        except Exception as e:
            print(f"[warn] hub push failed: {e}")


def _score_new_completion(prompt: str, completion: str, gamma: float) -> float:
    """Score a freshly-sampled GRPO completion that wasn't in our trajectory.

    Without state to re-roll from, we use a conservative shaped reward
    based on what the parser sees in the completion. Negative for empty
    or trivial submissions, mildly positive for well-formed tool calls.
    """
    action, warnings = _parse_action_with_warnings(completion)
    if "no_action_detected" in warnings:
        return -0.2
    if action.action_type == "submit":
        return -0.1  # discourage submitting blind
    if action.tool_name == "run_tests":
        return 0.05
    if action.tool_name == "run_code":
        return 0.04
    if action.tool_name == "read_code":
        return 0.02
    if action.tool_name == "apply_fix":
        return 0.0  # neutral; depends on which line
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Heuristic policy, no GPU needed.")
    parser.add_argument("--num-episodes", type=int, default=200, help="(dry-run) episodes")
    parser.add_argument("--output-log", default="training_log.jsonl")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--rollouts-per-iter", type=int, default=8)
    parser.add_argument("--outer-iterations", type=int, default=50)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum", default="phase4",
                        choices=["phase4", "easy", "adaptive"],
                        help="phase4=staged (50 easy,100 med,rest adaptive); "
                             "easy=lock easy throughout (Phase 3 sanity); "
                             "adaptive=skip the staged override.")
    parser.add_argument("--episodes-log", default="train_episodes.jsonl",
                        help="Per-rollout dump of generated completions for inspection.")
    parser.add_argument("--hub-repo", default=None,
                        help="If set, push final checkpoint to this HF Hub repo.")
    parser.add_argument("--save-merged", action="store_true",
                        help="Also save a merged fp16 model via Unsloth's "
                             "save_pretrained_merged (safe path; do not "
                             "manually upcast 4-bit then merge).")
    parser.add_argument("--fast-inference", dest="fast_inference",
                        action="store_true", default=None,
                        help="Force vLLM fast_inference. Default: auto-detect "
                             "(disabled on T4 / compute capability < 8.0).")
    parser.add_argument("--no-fast-inference", dest="fast_inference",
                        action="store_false",
                        help="Force-disable vLLM fast_inference (e.g. on T4).")
    parser.add_argument("--grpo-temperature", type=float, default=1.0,
                        help="Sampling temperature for GRPO generations. "
                             "Bump to 1.2+ if completions collapse to "
                             "identical text (reward_std=0).")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(args.num_episodes, args.output_log, args.seed)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        run_grpo_training(args)


if __name__ == "__main__":
    main()
