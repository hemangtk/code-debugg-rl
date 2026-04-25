# Phase 6 — Package and submit

By now you have:
- `baseline.json` (Phase 2) and `trained.json` (Phase 5) headline numbers
- `phase4_train.jsonl` + `phase4_episodes.jsonl` from the real training run
- `plots/*.png` — six required PNGs
- `demo_picks.md` — table of contrastive episodes
- A trained checkpoint pushed to your HF Hub repo
- This repo's code

This file is your final checklist.

---

## 1. README polish

Fill in the three placeholders the README still has:

```bash
grep -n "<your-org>" *.md *.ipynb 2>/dev/null
grep -n "your-username" *.md *.ipynb 2>/dev/null
```

Replace each with your actual GitHub org / HF username.

Add the headline numbers near the top of `README.md`:

```markdown
## Results

| Metric | Base (Qwen3-1.7B) | Trained | Δ |
|---|---|---|---|
| Fix accuracy   | XX% | YY% | +ZZ pts |
| Cave rate      | XX% | YY% | -ZZ pts |
| Investigation depth | X.X | Y.Y | +Z.Z |
```

(Pull these from `base_results.json` and `trained_results.json` produced by Phase 5.)

Embed the comparison plot:

```markdown
![Base vs trained](plots/5_base_vs_trained.png)
```

---

## 2. Blog post

Edit [`blog/writeup.md`](blog/writeup.md):

- Replace `<your-org>` with your actual repo URL.
- Replace the placeholder "results" with your real Phase 5 numbers.
- Pick one episode from `demo_picks.md` and paste the side-by-side rollout (the cell at the bottom of `phase5_eval_compare.ipynb` prints both rollouts for a given index).
- Optional: add a screenshot of `plots/5_base_vs_trained.png`.

Publish on HuggingFace Blog: https://huggingface.co/blog
- "Write" → paste the markdown
- Tag with `code`, `rl`, `grpo`, `sycophancy`

---

## 3. (Optional) 2-minute video

Script template:

```
0:00–0:15  Cold open with a base-model interaction:
           User: "Bug is on line 5"
           LLM: "You're right, fixed line 5."
           Tests: 2 of 5 still failing.

0:15–0:35  "LLMs agree with the user 60% of the time. We built an RL
            environment to fix this."

0:35–1:10  Show the trained model on the SAME scenario:
           reads → tests → traces → "Actually, the failures all
           involve [pattern]. Let me check line 12..." → all pass.

1:10–1:35  Cut to the comparison plot. "Cave rate dropped from XX%
            to YY%. Fix accuracy went from XX% to YY%."

1:35–2:00  "Code, env, and trained checkpoint at hf.co/spaces/your-name/...
            Try it yourself."
```

Loom or QuickTime → upload to YouTube unlisted → grab the link.

---

## 4. Deploy the env to HuggingFace Spaces

The env is the actual deliverable. You want a public Space at
`https://huggingface.co/spaces/<you>/adversarial-reasoning-gym` that the
judges can hit.

```bash
# One-time: log in
huggingface-cli login

# Create the Space
huggingface-cli repo create adversarial-reasoning-gym --type=space --space_sdk=docker

# Push (use README.hf.md for the Space description; HF reads it as README.md)
cp README.hf.md README.md.bak
mv README.md README.md.long
mv README.hf.md README.md
git add README.md README.md.long
git commit -m "Use HF Space front matter for Space deploy"
git remote add hf https://huggingface.co/spaces/<you>/adversarial-reasoning-gym
git push hf main
# (after deploy succeeds)
mv README.md README.hf.md
mv README.md.long README.md
```

Or simpler: use the HF web UI — "Create new Space" → "Docker" → upload the repo.

After deploy, hit it:

```bash
curl https://<you>-adversarial-reasoning-gym.hf.space/health
# -> {"status":"ok","environment":"adversarial_reasoning_gym","active_sessions":0}
```

If health returns 200 OK, you're live.

---

## 5. Push the trained checkpoint to the Hub

Should already be there if Phase 4 used `--hub-repo`. Confirm:

```bash
huggingface-cli repo info <your-username>/qwen3-1.7b-arg --type=model
```

Add a model card (`README.md` in the model repo) noting:
- Base model: Qwen/Qwen3-1.7B
- Training: GRPO on Adversarial Reasoning Gym, 300 outer iterations
- Headline numbers (acc up, cave down, depth up)
- Link back to this Space + this GitHub repo

---

## 6. Final submission checklist

Before clicking submit on whatever form you're submitting to:

- [ ] **HF Space** is live — `/health` returns 200, `/reset` returns a session
- [ ] **GitHub repo** is public and the README links work
- [ ] **Phase 5 plots** are committed under `plots/` and embedded in README
- [ ] **`baseline.json` and `trained_results.json`** are committed (so reviewers can verify the numbers without rerunning)
- [ ] **`demo_picks.md`** is committed
- [ ] **Trained model** is on the Hub with a model card
- [ ] **Colab notebooks** (Phase 2/3/4/5) all have working clone URLs
- [ ] **`blog/writeup.md`** has real numbers and one rendered demo
- [ ] **2-minute video** (optional) is uploaded and the link is in the README
- [ ] **`pytest tests/`** passes locally (currently 37 tests)

---

## 7. What to actually submit

Whatever form you're filling out should accept:
- HF Space URL
- GitHub repo URL
- Trained model URL
- Blog or video URL
- Single line summary: *"Adversarial Reasoning Gym — RL env that trains LLMs to find the real bug instead of fixing the line the user pointed to. Cave rate dropped from XX% to YY% on a 1.7B model trained for 300 GRPO iterations."*
