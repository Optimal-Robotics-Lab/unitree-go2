# Reward/Cost Audit — Consolidated Committee Report

**Subject:** `training/envs/unitree_go2/unitree_go2_joystick.py` (+ `config.py`, `base.py`)
**Inputs:** 4 independent sub-audits (tracking/orientation; effort/regularization; gait/contact; composition/config), cross-checked by this committee against the live source file.
**Method:** every code quote and line number below was re-verified against the current file on disk (line numbers as of this read: `step()` at 238-412, reward fns at 557-715, `RewardConfig` in `config.py` 11-42, `del` list in `base.py` 102-111). No stale references found — all four sub-reports' line numbers and code quotes matched the live source.

---

## 1. Executive Summary

Ranked by expected effect on training outcomes:

1. **Termination penalty is functionally neutered.** The `-1.0` `termination` weight cannot flip a good step negative (needs ≤ -2.49), and the `jnp.clip(..., 0.0, 10000.0)` floor erases the penalty's magnitude entirely on bad ("dirty") terminations, collapsing catastrophic falls and mild joint-limit trips to the identical `reward = 0.0`. This is the single highest-leverage fix — it affects the terminal-step signal of **every episode**, and mujoco_playground's own newer G1 task independently confirms and fixes this exact defect upstream.
2. **Gait coordination is effectively unenforced.** `_reward_air_time` scores a proper diagonal trot and a degenerate synchronized 4-foot hop identically (bit-for-bit equal reward distributions), and the term meant to catch that — `_cost_gait_variance` — is dead ~96% of the time and, when active, *rewards* the synchronized hop over the correct trot (inverted). Nothing in the current reward stack actually enforces alternating-gait behavior.
3. **`_reward_foot_clearance` has a free lunch on the high side.** `jnp.minimum(foot_z, target)` clamps the error to zero for any foot height at or above target, so a foot lifted to 1 m scores identically to one at exactly 0.1 m — no disincentive against excessive/wasteful lift height, unlike the (unclamped) reference implementation.
4. **`acceleration: -2.5e-7` is a no-op at any physically realistic `qacc`.** It would take ~18,900 rad/s² RMS per joint to matter — several orders of magnitude past even hard-impact transients. It isn't wrong, just decorative; a human should decide whether to retune it or drop it.
5. **`target_air_time = 0.5` is dead config** — parsed into `self.target_air_time` in `base.py` but never passed into `_reward_air_time`. Changing it in `config.py` has zero effect. `mode_time = 0.3` is the value actually governing reward-maximizing swing duration, and it plays a different role (hard cap vs. target).
6. Several **legitimate but debatable design choices** need a human call, not an auto-fix: `linear_z_velocity = -2.0` is 4x the mujoco_playground reference's `-0.5` (may be intentional retuning, may be a typo); the `del reward_config_dict[...]` sync between `RewardConfig`, `base.py`, and `step()`'s `rewards` dict is currently correct but hand-maintained with no assertion; `command_threshold = 0.0` with strict `>` works today only because commands are exact-multiplied by a boolean mask (fragile, no tolerance band, diverges from reference epsilons).
7. **Everything in the tracking/orientation/velocity-cost group is correct** (`_reward_tracking_velocity`, `_reward_tracking_yaw_rate`, `_cost_vertical_velocity`, `_cost_orientation_regularization`) — signs, frames, and magnitudes all check out against both mujoco_playground and IsaacLab conventions.
8. **Minor comment/scope mismatch:** `_cost_acceleration`'s comment says "Motor/Joint Acceleration" but it operates on the full `qacc` (18 DOF, including the 6 free-joint base DOFs), unlike `_cost_torques` which correctly restricts to the 12 actuated joints. Doc fix only, no behavior change needed.
9. Nothing found in this audit is causing outright training divergence — the environment will very likely still learn to walk, because tracking rewards dominate the sum by 2-3 orders of magnitude over most of the flagged issues. The two exceptions that plausibly *do* affect final policy quality are #1 (weak fall-avoidance/recovery-awareness signal) and #2 (no structural pressure toward a clean alternating gait — expect possible pacing/bounding/hopping gaits or asymmetric footfall patterns rather than a diagonal trot).

---

## 2. Confirmed Bugs

### 2.1 Termination cost is washed out by the reward-composition clip (highest priority)

**Location:** `step()`, `unitree_go2_joystick.py:359-362`; `_cost_termination`, line 714-715.

```python
rewards = {k: v * self.reward_config[k] for k, v in rewards.items()}
reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
```
`termination` weight = `-1.0` (`config.py:25`), `dt = 0.02`.

**Evidence (merged from two independent audits, same underlying arithmetic):**
- *Best case for the penalty to matter* — a "clean" termination where tracking is still near-perfect: max positive sum from tracking alone is `1.5·1 + 0.75·1 = 2.25`. To force the post-weight sum negative, `termination` would need to be ≤ `-2.49` (accounting for other small costs); the actual weight of `-1.0` is **2.5x too small** to guarantee a sign flip even here. Concretely: sum before dt ≈ `2.49 → 1.49` after termination fires → reward `0.0498 → 0.0298`. A 40% dip that reads to PPO as "a slightly worse ordinary step," not a terminal penalty.
- *Realistic case — a genuine fall*: orientation, angular-velocity, and unwanted-contact costs are already large and negative on the same post-physics `data` that triggered `done` (worked example: sum ≈ `-9.2` before dt, of which the `termination` term itself contributes only `-1.0` — a rounding error against the `-8.2` already present from instability costs). `jnp.clip(..., 0.0, ...)` floors this to **exactly 0.0** — indistinguishable from a much milder termination, and indistinguishable from what a `termination` weight of `-50` would also produce (both clip to 0.0). The clip destroys exactly the magnitude information needed on the step that matters most.
- **Upper bound of the clip (`10000.0`) is dead code** — max plausible pre-`dt` sum from all positive terms combined is ≈ 3.05, giving reward ≈ 0.061; the ceiling can never be reached. Confirmed unreachable, not a functional issue on its own, but signals this clip was never tuned for this task — it's a verbatim carry-over.
- **Cross-repo corroboration:** IsaacLab's `RewardManager.compute()` and mjlab's `reward_manager.py` both accumulate `value = func()*weight*dt` **with no clip at all** — negative step rewards propagate untouched, so a `-2.0`-class termination weight there directly manifests in the return. mujoco_playground's own Go1/Barkour/Spot/H1/OP3/T1/Berkeley-Humanoid joystick tasks all use this exact same `jnp.clip(..., 0.0, 10000.0)` + `termination=-1.0` pattern (so this is a verbatim inherited pattern, not a locally introduced bug) — **but mujoco_playground's newer `g1/joystick.py` (humanoid) removes the clip entirely (`reward = sum(rewards.values()) * self.dt`, no `jnp.clip`) and raises `termination` to `-100.0`**, two orders of magnitude above every other task. That is direct evidence from the same upstream team that they identified this exact interaction as a problem for tasks where clean fall/termination signal matters, and fixed it via (a) removing the floor and (b) drastically upweighting termination — not by tuning within the old scheme.

**Fix suggestion:** either (a) drop the lower clip bound (e.g. `jnp.clip(..., -100.0, 10000.0)` or no clip) so termination and instability costs retain their true magnitude on bad steps — the more complete fix, matching IsaacLab/mjlab/G1 — or (b), if a non-negative reward is required downstream, raise `termination` toward G1's `-100.0` scale *and* still relax the floor, since a floor at exactly `0.0` will keep collapsing all sufficiently-bad terminations to the same value regardless of the weight's magnitude.

### 2.2 Gait coordination is unenforced — two compounding defects

**2.2a — `_reward_air_time` has no cross-foot coupling.** (`unitree_go2_joystick.py:614-637`)

Purely per-foot; nothing references any other foot's phase. Measured reward distributions:
```
Normal trot (diagonal pairs)      : reward_mean = +0.3360
Degenerate in-phase 4-foot hop    : reward_mean = +0.3360   (bit-for-bit identical)
One foot permanently stuck stance : reward_mean = +0.2520
```
A proper trot and a degenerate synchronized hop are rewarded identically. IsaacLab's structural analog (`feet_air_time_positive_biped`) has a `single_stance` gate (`torch.sum(in_contact, dim=1) == 1`) that zeroes the reward when the wrong number of feet are down, plus `torch.min` instead of `sum` across feet so one badly-timed foot can't be masked by the others; mujoco_playground's Go1 avoids the problem differently, by firing the reward only once per foot at the `first_contact` event. This repo's port has neither mechanism.

**2.2b — `_cost_gait_variance` doesn't provide the missing coordination signal; it's dead most of the time and inverted when active.** (`unitree_go2_joystick.py:639-651`)

Tracing the exact state update in `step()` (269-280) against a simulated ideal trot (dt = 0.02s, 0.5s/0.5s duty cycle): `previous_air_time`/`previous_contact_time` are captured on the transition step, then **immediately overwritten back to 0.0 on the very next control step** (because the update condition — `feet_contacts == True` — stays true throughout stance). Nonzero for exactly 1 step per 25 (4% duty). Sampling 5 random post-warm-up instants of an ideal trot: `prev_air_time = [0,0,0,0]`, `var_cost = 0.0` every single time. When it does fire (at footfall transitions), a proper diagonal trot produces `[0.5, 0, 0, 0.5]` → nonzero variance (`cost_mean = 0.00500`, spiking to `0.125` at every footfall), while the degenerate in-phase 4-foot hop has all four feet transition simultaneously → `[0.5,0.5,0.5,0.5]` → **zero variance, always** (`cost_mean = 0.00000`), and a permanently-stuck-foot gait scores `cost_mean = 0.00406` — better than the correct trot. **This is the exact inverse of the intended ranking**: the worst gait (fully synchronized) scores best, the textbook-correct trot scores worst among the three.

Net effect: the codebase relies on `_cost_gait_variance` to supply the cross-foot coordination that `_reward_air_time` lacks, but that term is dead ~96% of the time and actively rewards the wrong thing in the remaining 4%. There is currently **no functioning mechanism** in this reward stack that prefers an alternating trot over a synchronized hop.

**Fix suggestion:** drop `_cost_gait_variance` in its current form. Add the missing coordination gate directly into `_reward_air_time` (IsaacLab's `single_stance`-count-gate pattern), or compute a variance/coordination signal over completed-cycle durations captured at `first_contact` events (mujoco_playground's pattern) rather than over a running array that's zero almost all the time.

### 2.3 `_reward_foot_clearance` — overshoot is free

**Location:** `unitree_go2_joystick.py:653-667`.
```python
foot_height = jnp.minimum(foot_position[..., -1], target_foot_height)
foot_error = jnp.square(foot_height - target_foot_height)
```
Measured: `foot_error = 0` for any `foot_z ≥ 0.1` (target), regardless of actual height — `foot_z = 1.0m` scores identically to `foot_z = 0.1m` at every tested xy velocity (0, 0.5, 2.0 m/s). Only undershoot is penalized. The mujoco_playground reference (`_cost_feet_clearance`) uses an unclamped `jnp.abs(foot_z - target) * vel_norm` — symmetric in both directions. No other cost term in this file provides meaningful counter-pressure against excessive lift height (torque/action-rate/acceleration weights are all shown elsewhere in this audit to be negligible at realistic magnitudes).

**Fix suggestion:** remove the `jnp.minimum` clamp — penalize `jnp.square(foot_position[..., -1] - target_foot_height)` directly, matching the reference's symmetric treatment.

### 2.4 `target_air_time` is dead configuration

**Location:** `config.py:33` (`target_air_time: float = 0.5`), `base.py:95` (`self.target_air_time = reward_config.target_air_time`), `base.py:104` (`del reward_config_dict['target_air_time']`).

Confirmed by grep across `training/envs/unitree_go2/`: `target_air_time` appears in exactly those three lines and is **never** passed into `_reward_air_time` (`unitree_go2_joystick.py:331-339`, whose signature has no `target_air_time` parameter at all). `mode_time = 0.3` is the value that actually determines the reward-maximizing swing/stance duration in `_reward_air_time`, and it plays a different role — a hard cap after which the reward stops increasing (verified: an absurd 10s-swing gait scores `reward_mean = +0.0088`, far below a natural trot's `+0.336`, so it doesn't create a "hold forever" incentive) — not a target duration to converge toward. Anyone tuning `target_air_time` to change gait cadence is tuning a knob with zero effect.

**Fix suggestion:** either wire it in (e.g. `-(duration - target_air_time)^2`, or reward proportional to `min(duration, target_air_time)`, matching mujoco_playground's `(air_time - target) * first_contact` pattern) or delete the field to stop implying it does something.

---

## 3. Design Concerns (human decision needed, not auto-fix)

- **`acceleration: -2.5e-7` weight is a no-op at all realistic `qacc` magnitudes.** To match the (already-small) `torque` term's contribution (~0.02-0.05/step) would require RMS per-DOF acceleration ≈ 18,900 rad/s² — several orders of magnitude beyond even a hard foot-strike transient (~1000 rad/s² generously); at that generous impact estimate the term still contributes only ≈0.0011, an order below `torque`. No reference implementation checked (mujoco_playground, IsaacLab) has a directly equivalent term at a comparable weight — mujoco_playground's closest analog (`energy: -0.001`) is ~4000x larger relative to its own scale. Not incorrect, just currently decorative. **Decision needed:** retune by 2-4 orders of magnitude (informed by an empirical `qacc` histogram from a rollout) or remove.
- **`linear_z_velocity = -2.0` is 4x the mujoco_playground Go1 reference (`-0.5`).** Every other shared weight (`angular_xy_velocity`, `torque`, `action_rate`, `stand_still`, `termination`, `foot_slip`) matches the reference exactly, and `tracking_linear_velocity`/`tracking_angular_velocity` are both scaled by a consistent 1.5x — but `linear_z_velocity` doesn't fit either pattern. Could be an intentional, considered retune (e.g. this robot bounces more) or a stray edit. **Decision needed:** confirm intentional.
- **`del reward_config_dict[...]` sync (`base.py:103-110`) has no runtime assertion.** Today it's an exact 1:1 match against the `rewards` dict built in `step()` and every `RewardConfig` field is accounted for (verified: 23 fields, 8 deleted as hyperparameters, 15 remain, all 15 match `step()`'s `rewards` keys exactly). But this is maintained purely by convention — a future `RewardConfig` field added without a corresponding `del` (or a `step()` key without a matching field) fails silently (extra key sits unused / seeds a spurious metrics key at `reset()`) or hard-crashes (`KeyError`) respectively, with no test catching it today. **Suggested fix (low-risk, not urgent):** one assertion in `base.py` after building `self.reward_config`, e.g. asserting the weight-field set and the known hyperparameter-field set partition `RewardConfig`'s fields.
- **`command_threshold = 0.0` with strict `>` is fragile, not incorrect.** Works today only because `sample_command()` builds commands via exact multiplication by a boolean mask (`stand_still_mask * command`), yielding an exact `0.0`, not noise-contaminated near-zero — no continuous noise is ever added to the command itself. Both reference implementations use a small positive epsilon instead (mujoco_playground: `0.01`; IsaacLab: `0.1`) specifically to avoid dependence on this kind of exact-zero invariant. **Decision needed:** low-cost hardening — set to `0.01`-`0.05` to remove the latent fragility, even though nothing is broken today.
- **`_cost_angular_velocity` uses world/global frame, causing minor yaw→roll/pitch kinematic leakage.** A robot with a nonzero pitch that is purely executing a commanded yaw turn (zero actual wobble) picks up a small nonzero cost purely from `ω_world = R·ω_body` leakage (measured: at 30° pitch, pure `wz=1.2 rad/s` yaw leaks to a pre-weight cost of 0.36, vs. 0.0 in body frame). At realistic trot pitch angles (a few degrees) this is ≲1e-5 per step after weighting — negligible next to tracking rewards (~0.02-0.03/step) — and mujoco_playground's Go1 has the identical construction and weight, so this is an inherited, low-impact design choice, not a locally introduced defect. **Optional fix:** swap to the already-computed body-frame gyro (`get_gyro(data)[:2]`) if tighter isolation is wanted; zero additional cost since the gyro reading already exists in `step()`.
- **`_cost_stand_still` has a hard discontinuity at `command_norm = 0.1`.** Verified numerically: cost jumps from `0.6` to exactly `0.0` crossing the threshold (representative pose error 0.05 rad × 12 joints). Standard for a gating term and matches the codebase's own style elsewhere (`_cost_foot_slip`, `_reward_air_time`) and upstream mujoco_playground's identical pattern (threshold `0.01` there vs. `0.1` here) — not a bug, but a coarser regularizer than a smooth gate would be. **Optional fix:** `jnp.exp(-command_norm² / tau)` if smoother pose-holding incentive near the threshold is desired.
- **`_cost_acceleration` comment/scope mismatch.** Comment says "Penalize Motor/Joint Acceleration" but operates on the full `qacc` (`nv=18`: 6 free-joint base DOFs + 12 actuated joints), unlike `_cost_torques` (`data.actuator_force`, correctly 12-actuated-joint-only). So the two terms presented in `config.py` as a matched "energy regularization" pair actually operate over different state scopes, undisclosed by the comment. **Fix:** either slice to `qacc[6:]` to match `_cost_torques`'s scope, or update the comment to disclose the base-DOF inclusion (e.g. useful for penalizing fast recovery/disturbance-response motions, if that's the intent).

---

## 4. Reviewed and Cleared

Checked directly against both code and reference implementations (IsaacLab, mjlab, mujoco_playground Go1/G1), no issue found:

- `_reward_tracking_velocity`, `_reward_tracking_yaw_rate` — correct shape, frame (body-frame, matches command convention), sign, and saturation behavior.
- `_cost_vertical_velocity`, `_cost_orientation_regularization` — correct sign/shape; global-frame choice for vz cost is deliberate and matches the mujoco_playground recipe this env is ported from (not an oversight vs. IsaacLab's body-frame convention — a documented design-family difference).
- Shared `kernel_sigma = 0.25` between the two tracking rewards — deliberately matched to `command_range = [1.5, 1.0, 1.2]`, mirroring mujoco_playground Go1's `a=[1.5, 0.8, 1.2]` + identical sigma; produces informative (non-saturated) rewards across realistic tracking-error ranges.
- `_cost_torques` (L1+L2 combination) — coherent "elastic net"-style design (sparsity + magnitude), not double-counting; negligible in absolute terms (~1e-3/step) regardless.
- `_cost_action_rate`, `_cost_unwanted_contact` — straightforward, correctly scaled, no sign or discontinuity issues.
- `t_max`/`t_min` mutual exclusivity between `air_time` and `contact_time` — confirmed complementary gating, 0 violations across 1000 simulated steps.
- `mode_time` "hold forever" perverse incentive — does not materialize; reward saturates then drops to exactly 0 past `mode_time`.
- `_cost_foot_slip` (active implementation) — near-line-for-line match to mujoco_playground's `_cost_feet_slip`; the commented-out height-gated alternative reads as an abandoned experiment, not evidence against the active version.
- `del reward_config_dict[...]` (`base.py`) vs. `RewardConfig` fields vs. `step()`'s `rewards` dict — exact 1:1 match today, verified by full enumeration on both sides (see Design Concerns for the missing-assertion caveat).
- `get_observation` docstring vs. actual concatenation order — matches exactly.

---

## 5. Prioritized Action List

1. **Fix the termination/clip interaction (§2.1).** Touches the terminal-step reward of every single episode; corroborated by upstream G1's independent fix. Start here — it's a two-line change (relax or remove the clip floor, and/or raise `termination` toward the G1-precedent scale) with the highest signal-to-effort ratio in this audit.
2. **Fix gait coordination (§2.2a/2.2b) — treat as one change.** Remove or replace `_cost_gait_variance`, and add a coordination gate to `_reward_air_time` (IsaacLab's `single_stance`-style approach is the more minimal patch given the existing per-foot structure). This is the second-most consequential fix because it's currently shaping *zero* preference for a proper gait over a degenerate one — likely to matter for what gait actually emerges, not just fine-tuning.
3. **Fix `_reward_foot_clearance` overshoot (§2.3).** Simple, well-evidenced, low-risk (remove one `jnp.minimum` clamp) — do this alongside #2 since it's in the same reward family and touches gait quality.
4. **Resolve `target_air_time` dead config (§2.4).** Either wire it in or delete it — cheap, prevents future confusion, no urgency.
5. **Human decisions on the three tuning/config concerns (§3, first three items):** confirm `acceleration` weight intent, confirm `linear_z_velocity` 4x-reference intent, and decide whether to add the `del`-list assertion. None of these are urgent, but they're cheap and should be resolved before the next long training run to avoid re-diagnosing them later.
6. **Everything else in §3** (comment fixes, `command_threshold` epsilon, angular-velocity frame, stand-still smoothing) — low priority, address opportunistically or bundle into a cleanup pass; none block a training run.
