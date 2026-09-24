"""Evaluates candidate Spot leg PD gains by standing the robot at zero action.

Usage:
    uv run python tuning/pd_design.py --kp 60,80,152 [--zeta 0.7] [--rho 0.75]
    uv run python tuning/pd_design.py --sweep_thigh 34,55,65,80,100

For each candidate this prints the standing torso height, the worst-leg sag
per joint type, and the lowest torso height under random reset offsets (a
robust stance stays high; a marginal one folds). It also derives, per joint
type:
  kv           = 2 * zeta * sqrt(kp * inertia)   damping ratio `zeta`
  action_scale = rho * torque_limit / kp         a full-range action commands
                                                 `rho` of the torque limit
Static holding torque is not used to size kp: the thigh's load grows as the
legs fold, so its sag is bistable and a torque/sag formula badly
underestimates the gain it needs.
"""

import argparse
import pathlib

import mujoco
import numpy as np

_SCENE = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'training/envs/spot/mjcf/scene_mjx_simplified.xml'
)
_NUM_LEGS = 4
_STAND_SECONDS = 2.5
_RESET_NOISE = (0.05, 0.1)  # [rad], joint-offset draws
_NOISE_SEEDS = 3
_TYPES = ('abduction', 'thigh', 'calf')


def joint_inertia(model: mujoco.MjModel) -> np.ndarray:
    """Returns worst-leg diagonal joint-space inertia [kg m^2] per type."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.key('home').qpos
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    return np.diag(mass)[6:18].reshape(_NUM_LEGS, 3).max(axis=0)


def stand(
    model: mujoco.MjModel,
    kp: np.ndarray,
    kv: np.ndarray,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Stands at zero action; returns torso height [m] and joint sag [4, 3] [rad]."""
    default_qpos = model.key('home').qpos.copy()
    data = mujoco.MjData(model)
    data.qpos[:] = default_qpos
    rng = np.random.default_rng(seed)
    data.qpos[7:19] += rng.uniform(-noise, noise, 12)
    mujoco.mj_forward(model, data)
    limits = model.actuator_forcerange
    # Legs use the candidate gains; the arm holds its pose with stiff gains.
    kp_all = np.concatenate([np.tile(kp, _NUM_LEGS), np.full(model.nu - 12, 200.0)])
    kv_all = np.concatenate([np.tile(kv, _NUM_LEGS), np.full(model.nu - 12, 10.0)])
    for _ in range(int(_STAND_SECONDS / model.opt.timestep)):
        torque = kp_all * (default_qpos[7:] - data.qpos[7:]) - kv_all * data.qvel[6:]
        data.ctrl[:] = np.clip(torque, limits[:, 0], limits[:, 1])
        mujoco.mj_step(model, data)
    return data.qpos[2], (data.qpos[7:19] - default_qpos[7:19]).reshape(_NUM_LEGS, 3)


def evaluate(model, kp, zeta, rho, inertia, torque_limit) -> None:
    kv = 2.0 * zeta * np.sqrt(kp * inertia)
    scale = rho * torque_limit / kp
    height, sag = stand(model, kp, kv)
    noisy = [
        stand(model, kp, kv, noise, seed)[0]
        for noise in _RESET_NOISE for seed in range(_NOISE_SEEDS)
    ]
    print(f'kp {kp.round(0).tolist()} kv {kv.round(1).tolist()} '
          f'action_scale {scale.round(2).tolist()}')
    print(f'    torso z {height:.3f} m | worst sag (abd, thigh, calf) '
          f'{np.abs(sag).max(axis=0).round(2)} rad | min z under reset noise '
          f'{min(noisy):.3f} m | full-range torque {(kp * scale).round(0)} Nm')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kp', type=str, default='60,80,152')
    parser.add_argument('--sweep_thigh', type=str, default='')
    parser.add_argument('--zeta', type=float, default=0.7)
    parser.add_argument('--rho', type=float, default=0.75)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    inertia = joint_inertia(model)
    torque_limit = model.actuator_forcerange[:3, 1]
    base_kp = np.array([float(x) for x in args.kp.split(',')])
    candidates = [base_kp]
    if args.sweep_thigh:
        candidates = [
            np.array([base_kp[0], float(k), base_kp[2]])
            for k in args.sweep_thigh.split(',')
        ]
    for kp in candidates:
        evaluate(model, kp, args.zeta, args.rho, inertia, torque_limit)


if __name__ == '__main__':
    main()
