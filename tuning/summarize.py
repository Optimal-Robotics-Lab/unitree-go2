"""Summarizes one training run's --metrics_file for the reward-tuning loop.

Usage:
    uv run python tuning/summarize.py tuning/runs/<trial>.jsonl [--last 3]

Prints a JSON object with the run status, the scorecard averaged over the
last N evaluations (all `score/*` keys, weight-independent), and the last
evaluation's weighted per-term breakdown (`eval/episode_<term>`) for
diagnosing which reward terms are driving behavior.
"""

import argparse
import json
import math

_DEFAULT_FINAL_ITERATION = 20  # spot_train.py: 20 epochs for a full run.
_MIN_HEALTHY_EPISODE_LENGTH = 300


def _finite(value: float | None) -> bool:
    return value is None or math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_file')
    parser.add_argument('--last', type=int, default=3)
    parser.add_argument(
        '--final_iteration', type=int, default=_DEFAULT_FINAL_ITERATION,
        help='Iteration at which the run counts as complete (its --num_epochs).',
    )
    args = parser.parse_args()

    with open(args.metrics_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        print(json.dumps({'status': 'no_data'}))
        return

    last = records[-1]
    non_finite = sorted({
        key for record in records for key, value in record.items()
        if isinstance(value, float) and not _finite(value)
    })
    collapsed = (
        last['iteration'] >= 10
        and last['score/episode_length_steps'] < _MIN_HEALTHY_EPISODE_LENGTH
    )
    if non_finite or collapsed:
        status = 'hard_failure'
    elif last['iteration'] >= args.final_iteration:
        status = 'complete'
    else:
        status = 'running'

    window = [r for r in records if r['iteration'] >= 1][-args.last:]
    scorecard = {}
    for key in last:
        if not key.startswith('score/'):
            continue
        values = [r[key] for r in window if r.get(key) is not None]
        scorecard[key] = sum(values) / len(values) if values else None

    print(json.dumps({
        'status': status,
        'iteration': last['iteration'],
        'non_finite_keys': non_finite,
        'episode_length_collapsed': collapsed,
        'scorecard_mean_of_last': {'n': len(window), **scorecard},
        'weighted_term_breakdown': {
            key.removeprefix('eval/episode_'): value
            for key, value in last.items()
            if key.startswith('eval/episode_')
            and not key.endswith('_std')
            and not key.startswith('eval/episode_diag_')
        },
    }, indent=2))


if __name__ == '__main__':
    main()
