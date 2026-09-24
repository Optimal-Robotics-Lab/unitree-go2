# Reward presets

Presets are reward overrides for setups that diverge from the baseline reward
in `spot_train.py` (which is also the default training reward: `action_rate`
-0.2, `foot_clearance_sigma` 0.01). Apply with `--reward_preset=<name>`;
`--reward_overrides` on the command line is merged on top and wins.

| Preset | Use | Notes |
|--------|-----|-------|
| `finetune_energy_v1` | finetune from a checkpoint trained on the baseline reward | `action_rate` -0.5, `torque` -1e-3, `acceleration` -1e-3 (5x the earlier -2e-4). |

Finetune example (restores from the `true-dragon-17` checkpoint, the v3
no-filter run trained on the baseline reward):

    uv run python spot_train.py --tag=<tag> --restore_run=true-dragon-17 \
      --num_epochs=10 --reward_preset=finetune_energy_v1

## Tested finetunes (5 epochs from `true-dragon-17`, last-3-eval means)

| Overrides | Result |
|-----------|--------|
| `action_rate` -0.2, `torque` -1e-3, `acceleration` -2e-4 | Cost of transport 0.615 -> 0.592, longer stride, cleaner trot; action deltas -4-5%. |
| `action_rate` -0.5, same torque/acceleration | Best: action deltas -15-21%, stride 0.388 m, cost of transport 0.586. |
| above + `stand_still` -3.0 | Worse: stand-still joint velocity 0.75, stand-still action delta back to 0.169. Keep `stand_still` at -1.0. |

Stand-still joint velocity stayed about 0.70 rad/s in every variant.
