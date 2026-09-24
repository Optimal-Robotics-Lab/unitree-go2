# Spot reward-tuning loop prompt

Use with `/loop` (dynamic pacing). Paste everything below the line.

---

You are tuning the reward weights and hyperparameters of the Spot (with arm)
locomotion policy in this repo. Work in small, verifiable steps. You are
stateless between ticks: read `tuning/STATE.md` first every tick and update it
last.

## Goal
Find reward overrides that give a clearly better 20-epoch policy than the
baseline, as measured by the fixed scorecard, not by the training reward.
"Better" means: better velocity tracking, higher foot clearance, minimal foot
slip, no pronking/bounding, and smooth (low-jitter) control even when the
command is to stand still. The rules for what counts are in
`tuning/GATES.md`. You must not edit it.

## Why not just maximize the reward
The training reward is the thing you are allowed to change, so it cannot also
be the score. Never judge a trial on `eval/episode_reward` or on any
`eval/episode_<term>` value (those are already multiplied by the weights you
change). Judge only on the `score/*` values from `tuning/summarize.py`.

## What you may change
- ONLY the `--reward_overrides` JSON (fields of `RewardWeights` and
  `RewardHyperparameters` in `training/envs/spot/config.py`; the baseline
  values are written out in `spot_train.py`) and `--seed`.
- Every weight stays within [x0.25, x4] of its baseline value and is never 0
  and never sign-flipped. `termination` is frozen. Do not change batch size,
  rollout length, num_envs, epochs, or any file outside `tuning/`.
- Vary ONE term (or one tightly coupled pair) per trial, except when combining
  changes that were each already accepted.

## Running a trial (one at a time -- never run two trainings concurrently)
```
uv run python -u spot_train.py --tag=<trial_tag> --seed=<seed> \
  --reward_overrides='<json>' \
  --metrics_file=tuning/runs/<trial_tag>.jsonl \
  > tuning/runs/<trial_tag>.log 2>&1 &
```
- Do NOT set `WANDB_MODE`; runs must log online so the human can watch the
  `Visualizer` HTML for every epoch. Grab the run URL from the log
  (`wandb: View run at ...`) and put it in STATE.md.
- A full run is 20 epochs at 8192 envs: about 10 minutes, about 6 GB of GPU
  memory. Record the PID in STATE.md under "Run in flight".
- Score a run with `uv run python tuning/summarize.py tuning/runs/<tag>.jsonl`.
  It reports `status`: `running`, `complete`, or `hard_failure`.

## Abort policy (hard failures only)
Kill the run (by PID) if `summarize.py` reports `hard_failure` (NaN/inf in any
metric, or episode length collapsed below 300 at iteration >= 10) or the log
shows a crash. Do NOT abort a run because its scorecard looks worse at low
iterations: early behavior is exploration and says nothing about the final
policy. Record aborts and their reason in STATE.md.

## Batch 0: noise floor
Run the unmodified baseline with seeds 42, 43 and 44. From the 3 final
scorecards compute the per-metric std (sigma) and write it in STATE.md. Every
"beats by 2 sigma" test uses this. If the seeds differ so much that no
plausible change could clear 2 sigma on tracking, stop and report it.

## Later trials
1. Read the last completed run's `weighted_term_breakdown` and scorecard,
   compare to the gates, and pick the metric furthest from its gate.
2. Write the hypothesis in STATE.md before launching ("foot_slip 0.09 vs gate
   0.05 -> raise |foot_slip| weight x2 should cut slip").
3. Launch, wait, score, and record a verdict: ACCEPT (all gates hold and >=1
   improve metric beats the current best by >2 sigma), REJECT, or INCONCLUSIVE.
   On ACCEPT, update "Current best overrides" (later trials build on it).
4. If a change trades one gate for another, record it as a lesson and do not
   accept it.

## Pacing
Use ScheduleWakeup. After launching a run, first check about 9 minutes later,
then every ~2 minutes until it finishes. Do not poll faster.

## Stop and report when
- 20 trials are done, OR
- 3 consecutive trials give no ACCEPT, OR
- a candidate passes every gate with room to spare.
Then stop the loop and write a summary: the best overrides, its scorecard
against the baseline, the wandb URLs of the baseline and the best run, and
what you learned. The human reviews the videos before anything is adopted.

## If something looks wrong
Crashes, every trial degenerate, GPU out of memory, metrics all null, or
surprising behavior: stop the loop and report. Do not improvise fixes.
Never touch git, PLAN.md, or anything outside `tuning/`.
