"""
    Motor Model / Actuation Pipeline:

    Composable JAX actuator torque model, replacing MuJoCo's `<motor>`
    actuators for direct-torque joints with an explicit per-substep
    pipeline: a plain tuple of `Stage`s, each reading and returning the full
    `ActuationPipelineState`.

    Only valid for actuators MuJoCo treats as direct-torque (`<motor>`).
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx

from training.envs.spot import transmission_constants as tc
from training.envs.spot.delay import DelayLine


def resolve_actuator_ids(
    mj_model: mujoco.MjModel, actuator_names: list[str],
) -> tuple[jax.Array, jax.Array]:
    """Resolves each named actuator's joint to `qpos`/`qvel` indices.

    Goes through `actuator_trnid` instead of assuming a fixed offset, so it
    holds for any joint order.

    Args:
        mj_model: Model to resolve indices against.
        actuator_names: Actuator names; their order sets the output order
            and must match the setpoint column order.

    Returns:
        `(qpos_ids, qvel_ids)`, each shape `(len(actuator_names),)`.
    """
    qpos_ids = []
    qvel_ids = []
    for name in actuator_names:
        actuator_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id == -1:
            raise ValueError(f"Actuator '{name}' not found in model.")
        joint_id = mj_model.actuator_trnid[actuator_id, 0]
        qpos_ids.append(int(mj_model.jnt_qposadr[joint_id]))
        qvel_ids.append(int(mj_model.jnt_dofadr[joint_id]))
    return jnp.array(qpos_ids), jnp.array(qvel_ids)


@struct.dataclass
class MotorModel:
    """Per-actuator PD gains and torque-speed saturation limits.

    Unlike Unitree Go2's `MotorModel`, this doesn't decompose `tau_max`/
    `omega_max` into a motor-shaft spec times a gear ratio -- we have no
    sourced per-motor/gearbox split for Spot, only the resulting envelope
    (if and when one is sourced). Revisit if that changes.

    Attributes:
        qpos_ids: Each actuator's joint index into `data.qpos`, shape
            `[n_actuators]`.
        qvel_ids: Each actuator's joint index into `data.qvel`, same shape.
        kp: Position gain [N*m/rad]; scalar or shape `[n_actuators]`.
        kv: Velocity gain [N*m*s/rad]; same shape options as `kp`.
        tau_max: Stall torque at zero speed [N*m], shape `[n_actuators]`, or
            `None` if unsourced -- `torque_speed_saturate` isn't usable then,
            so leave it out of the stage tuple.
        omega_max: No-load speed [rad/s], shape `[n_actuators]`, or `None`.
    """
    qpos_ids: jax.Array
    qvel_ids: jax.Array
    kp: float | jax.Array
    kv: float | jax.Array
    tau_max: jax.Array | None = None
    omega_max: jax.Array | None = None

    @property
    def damping_slope(self) -> jax.Array:
        """Torque-speed envelope's slope [N*m*s/rad]."""
        return self.tau_max / self.omega_max


@struct.dataclass
class ActuationPipelineState:
    """Everything a `Stage` might read or write. New fields never change an
    existing stage's signature.

    `data`, the setpoints, `feedforward_torque`, and `torque` are ephemeral:
    reconstructed fresh for every substep. `delay_state` is the exception --
    it must survive across substeps (and across `env.step()` calls, via
    `state.info`), since it's a history of past setpoints.

    Attributes:
        data: This substep's physics state; source of true joint state.
        motor_model: Per-actuator PD gains and torque-speed limits.
        qpos_setpoint: Commanded joint position [rad]. Stages may overwrite
            it in place (e.g. `delay.delay_stage` delays it before the
            control law reads it).
        qvel_setpoint: Commanded joint velocity [rad/s].
        feedforward_torque: Torque added on top of the control law [N*m].
        torque: Torque from the stages run so far [N*m].
        delay: Per-channel actuation delay [s].
        use_interp: Per-channel delay-read mode; `True` reads the smooth
            fractional interpolation, `False` the nearest-substep discrete
            lag. See `delay.DelayLine.read`.
        delay_state: Setpoint history buffer read by the delay stage.
    """
    data: mjx.Data
    motor_model: MotorModel
    qpos_setpoint: jax.Array
    qvel_setpoint: jax.Array
    feedforward_torque: jax.Array
    torque: jax.Array
    delay: jax.Array
    use_interp: jax.Array
    delay_state: DelayLine.State


# A stage receives and returns the full pipeline state. It may ignore most
# fields -- a control law only touches `torque`, a saturation stage only
# reads `torque` and clips it -- but a stateful stage (delay) can also
# update its own field in the same return. Stages compose as a plain tuple
# of functions.
Stage = Callable[[ActuationPipelineState], ActuationPipelineState]


def control_law(state: ActuationPipelineState) -> ActuationPipelineState:
    """PD law plus feedforward; ignores the incoming `state.torque`."""
    actuator_qpos = state.data.qpos[state.motor_model.qpos_ids]
    actuator_qvel = state.data.qvel[state.motor_model.qvel_ids]
    position_error = state.qpos_setpoint - actuator_qpos
    velocity_error = state.qvel_setpoint - actuator_qvel
    torque = (
        state.feedforward_torque
        + state.motor_model.kp * position_error
        + state.motor_model.kv * velocity_error
    )
    return state.replace(torque=torque)


def torque_speed_saturate(state: ActuationPipelineState) -> ActuationPipelineState:
    """Clips `state.torque` to a linear torque-speed envelope.

    Available torque falls from `tau_max` at zero speed to 0 at `omega_max`,
    symmetric in direction. Reads true joint velocity (`state.data.qvel`),
    not a setpoint -- the physical envelope depends on the actual rotor
    speed regardless of what's commanded.
    """
    actuator_qvel = state.data.qvel[state.motor_model.qvel_ids]
    available = jnp.clip(
        state.motor_model.tau_max
        - state.motor_model.damping_slope * jnp.abs(actuator_qvel),
        0.0,
        state.motor_model.tau_max,
    )
    torque = jnp.clip(state.torque, -available, available)
    return state.replace(torque=torque)


def knee_torque_limit_stage(knee_indices: jax.Array) -> Stage:
    """Builds a stage clipping the knee actuators' torque to Boston Dynamics'
    published position-dependent envelope
    (`transmission_constants.knee_max_torque`).

    Stands in for `torque_speed_saturate` on the knees specifically: their
    variable-ratio linkage makes peak torque a function of joint angle, not
    speed, and that curve is sourced (unlike a generic torque-speed
    envelope, which isn't). Leaves every other actuator's torque untouched.

    Args:
        knee_indices: Indices of the 4 knee actuators into the pipeline's
            actuator ordering (`motor_model.qpos_ids`/`state.torque`).
    """

    def stage(state: ActuationPipelineState) -> ActuationPipelineState:
        knee_angle = state.data.qpos[state.motor_model.qpos_ids[knee_indices]]
        max_torque = tc.knee_max_torque(knee_angle)
        knee_torque = jnp.clip(state.torque[knee_indices], -max_torque, max_torque)
        torque = state.torque.at[knee_indices].set(knee_torque)
        return state.replace(torque=torque)

    return stage


def delay_stage(delay_line: DelayLine) -> Stage:
    """Builds a stage that delays `qpos_setpoint` through `delay_line`.

    `delay_line` is closed over because it's structural (channel count,
    substep period, max representable delay) and never varies per episode --
    unlike the per-channel `delay`/`use_interp` values, which live on
    `ActuationPipelineState` so they can be domain-randomized.

    Args:
        delay_line: Setpoint history buffer; must have been sized with
            `num_channels` matching `qpos_setpoint`'s length.
    """

    def stage(state: ActuationPipelineState) -> ActuationPipelineState:
        delayed_qpos, delay_state = delay_line.step(
            state.delay_state, state.qpos_setpoint, state.delay, state.use_interp,
        )
        return state.replace(qpos_setpoint=delayed_qpos, delay_state=delay_state)

    return stage


def build_actuation_function(
    stages: tuple[Stage, ...] = (control_law, torque_speed_saturate),
) -> Callable[[ActuationPipelineState], ActuationPipelineState]:
    """Builds the per-substep torque pipeline from `stages`, run in order.

    `motor_model` (and `delay`/`use_interp`) live on the pipeline state
    rather than being closed over here, the same way `model` is an argument
    to `mjx.step` -- so domain-randomized per-episode values (see
    `randomize.py`) flow through without rebuilding the pipeline.

    Args:
        stages: Run in order; prepend `delay_stage(...)` to delay the
            setpoint before the control law, append a stage to add an
            effect after saturation, or replace `stages[0]` to swap the
            control law entirely.
    """

    def actuation_function(
        state: ActuationPipelineState,
    ) -> ActuationPipelineState:
        """Runs every stage and returns the state with the final torque."""
        state = state.replace(torque=jnp.zeros_like(state.feedforward_torque))
        for stage in stages:
            state = stage(state)
        return state

    return actuation_function
