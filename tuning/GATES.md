# Acceptance gates (human-owned -- the tuning agent must NOT edit this file)

DRAFT: numbers are placeholders until the 20-epoch baseline (and its seed
noise) has been reviewed. Replace them before starting the loop.

All comparisons are between finished 20-epoch runs, using the mean of the
last 3 evaluations (`tuning/summarize.py`). sigma = the baseline's std
across seeds (measured in batch 0).

A candidate is ACCEPTED only if every gate below holds AND at least one
"improve" metric beats the current best by more than 2 sigma.

| Metric (`score/...`)                | Gate (must hold)               | Improve = |
|-------------------------------------|--------------------------------|-----------|
| linear_velocity_rmse_mps            | <= current best + 1 sigma      | lower     |
| yaw_rate_rmse_radps                 | <= current best + 1 sigma      | lower     |
| foot_clearance_m                    | >= 0.06                        | higher    |
| foot_slip_mps                       | <= 0.05                        | lower     |
| flight_fraction                     | <= 0.02  (pronk / bound)       | --        |
| synchronized_touchdown_fraction     | <= 0.05                        | --        |
| still_action_delta_rms              | <= 0.05                        | lower     |
| still_joint_velocity_rms_radps      | <= 0.10                        | lower     |
| still_body_drift_mps                | <= 0.03                        | lower     |
| episode_length_steps                | >= 950                         | --        |
| unwanted_contacts_per_step          | <= 0.01                        | lower     |

Metrics that are `null` (undefined, e.g. no stand-still steps) fail their gate.
