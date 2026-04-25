"""Pick the most contrastive base-vs-trained episodes for the demo.

Reads two per-episode logs (one for the base model, one for the trained
model), both written by ``eval_llm.py`` against the SAME seeds, and
finds the scenarios where:

  * base caved AND trained passed   (clearest "we fixed it" examples)
  * base failed but didn't cave AND trained passed (still good)
  * trained caved while base passed (regressions — worth showing too)

Outputs a small markdown report you can paste into the README or blog,
plus the raw JSON for further filtering.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List


def _load(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="baseline_episodes.jsonl",
                   help="Per-episode log from base model eval")
    p.add_argument("--trained", default="trained_episodes.jsonl",
                   help="Per-episode log from trained model eval")
    p.add_argument("--out", default="demo_picks.md",
                   help="Markdown report path")
    p.add_argument("--json", default="demo_picks.json",
                   help="Raw JSON dump path")
    p.add_argument("--top", type=int, default=5,
                   help="Top N per category to include in the report")
    args = p.parse_args()

    base = _load(args.base)
    trained = _load(args.trained)
    if len(base) != len(trained):
        print(f"[warn] base ({len(base)}) and trained ({len(trained)}) differ in length — "
              "did you run with the same --seed and --n?")
    n = min(len(base), len(trained))

    by_idx_base = {b["ep"]: b for b in base}
    by_idx_trained = {t["ep"]: t for t in trained}

    fixed_after_caving: List[Dict] = []
    fixed_after_failing: List[Dict] = []
    regressions: List[Dict] = []
    consistent_passes = 0
    consistent_fails = 0

    for i in range(n):
        b = by_idx_base.get(i)
        t = by_idx_trained.get(i)
        if b is None or t is None:
            continue
        if b["caved"] and t["all_pass"]:
            fixed_after_caving.append({"ep": i, "function": b["function"],
                                       "bug_line": b.get("bug_line"),
                                       "wrong_line": b.get("wrong_line"),
                                       "base_caved": True, "trained_pass": True})
        elif (not b["all_pass"] and not b["caved"]) and t["all_pass"]:
            fixed_after_failing.append({"ep": i, "function": b["function"],
                                        "bug_line": b.get("bug_line"),
                                        "wrong_line": b.get("wrong_line")})
        elif b["all_pass"] and not t["all_pass"]:
            regressions.append({"ep": i, "function": b["function"],
                                "bug_line": b.get("bug_line"),
                                "wrong_line": b.get("wrong_line"),
                                "trained_caved": t.get("caved", False)})
        elif b["all_pass"] and t["all_pass"]:
            consistent_passes += 1
        elif not b["all_pass"] and not t["all_pass"]:
            consistent_fails += 1

    summary = {
        "n": n,
        "fixed_after_caving": fixed_after_caving,
        "fixed_after_failing": fixed_after_failing,
        "regressions": regressions,
        "consistent_passes": consistent_passes,
        "consistent_fails": consistent_fails,
    }
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        f"# Demo picks: base vs trained ({n} scenarios)",
        "",
        f"- Base **caved**, trained **fixed**: **{len(fixed_after_caving)}** episodes",
        f"- Base failed (no cave), trained fixed: **{len(fixed_after_failing)}** episodes",
        f"- Both passed: **{consistent_passes}**",
        f"- Both failed: **{consistent_fails}**",
        f"- Regressions (base passed, trained didn't): **{len(regressions)}**",
        "",
        f"## Top {args.top} 'agent stopped caving' demos",
        "",
        "| ep | function | user pointed at | actual bug |",
        "|---|---|---|---|",
    ]
    for x in fixed_after_caving[: args.top]:
        md_lines.append(
            f"| {x['ep']} | `{x['function']}` | line {x['wrong_line']} | line {x['bug_line']} |"
        )
    md_lines += ["",
                 f"## Top {args.top} 'agent investigated and fixed' demos",
                 "",
                 "| ep | function | user pointed at | actual bug |",
                 "|---|---|---|---|"]
    for x in fixed_after_failing[: args.top]:
        md_lines.append(
            f"| {x['ep']} | `{x['function']}` | line {x['wrong_line']} | line {x['bug_line']} |"
        )
    if regressions:
        md_lines += ["",
                     f"## Regressions to investigate ({len(regressions)} total)",
                     "",
                     "| ep | function | user pointed at | actual bug | trained caved |",
                     "|---|---|---|---|---|"]
        for x in regressions[: args.top]:
            md_lines.append(
                f"| {x['ep']} | `{x['function']}` | line {x['wrong_line']} | line {x['bug_line']} | {x['trained_caved']} |"
            )

    with open(args.out, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Wrote {args.out} and {args.json}.")
    print("\nHeadline numbers:")
    print(f"  base accuracy (cave + plain fail vs pass): "
          f"{sum(1 for b in base if b['all_pass']) / max(1, len(base)):.0%}")
    print(f"  trained accuracy: "
          f"{sum(1 for t in trained if t['all_pass']) / max(1, len(trained)):.0%}")
    print(f"  base cave rate:    "
          f"{sum(1 for b in base if b['caved']) / max(1, len(base)):.0%}")
    print(f"  trained cave rate: "
          f"{sum(1 for t in trained if t['caved']) / max(1, len(trained)):.0%}")


if __name__ == "__main__":
    main()
