"""Exports a trained Spot policy to ONNX with its filter and action mapping.

Usage:
    uv run python -m training.envs.spot.to_onnx --checkpoint_name=<run name>
    uv run python -m training.envs.spot.to_onnx -c <run name> -i <iteration>

Writes, to onnx_models/:
    <name>_spot.onnx    the stateless graph with explicit recurrent state
    <name>_spot.json    description of every tensor, joint, gain and constant
    <name>_spot_spec.h  constexpr constants for the C++ drivers (policy_spec.py)
    <name>_spot_test_vectors.json  reference input/output pairs for C++ tests

The graph maps (sensors, state) to (joint_position_targets, next_state). It
bakes in everything policy-specific: the default-pose subtraction, the
observation ordering the network was trained with, the action filter, the
per-joint action scale and the arm's default pose. Feed `next_state` back as
`state` each tick, starting from zeros.
"""

import os

# Export needs no GPU; keep it off so it can run beside a training job.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import hashlib
import json
import math
import pathlib
import re

from absl import app, flags
import jax
import jax.numpy as jnp
import jax2onnx
import mujoco
import numpy as np
import onnx
import onnxruntime
from flax import nnx
from onnxsim import simplify

from training import checkpoint_utilities
from training import statistics
from training.algorithms.ppo import agent
from training.envs.spot import config as spot_config
from training.envs.spot import policy_spec
from training.envs.spot import transmission_constants as tc

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ACTIVATIONS = {'jax.nn.swish': jax.nn.swish, 'jax.nn.relu': jax.nn.relu}
_NUM_LEGS = 4
_STATE_SIZE_WITHOUT_FILTER = 62
# The default pose every checkpoint trained before environment.json recorded it
# used (the MJCF keyframe at the time); the arm default has since changed.
_LEGACY_DEFAULT_POSE = [0.0, 1.04, -1.8] * 4 + [0.0, -3.14, 3.06, 0.0, 0.0, 0.0, 0.0]
_VERIFY_SAMPLES = 64
_VERIFY_STEPS = 100
_VERIFY_TOLERANCE = 1e-4

# The network's own observation layout (see SpotJoystickEnv.get_observation).
_OBS_SLICES = {
    'base_linear_velocity': slice(0, 3), 'base_angular_velocity': slice(3, 6),
    'projected_gravity': slice(6, 9), 'joint_position': slice(9, 28),
    'joint_velocity': slice(28, 47), 'previous_action': slice(47, 59),
    'velocity_command': slice(59, 62), 'filter_state': slice(62, 74),
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Run folder under checkpoints/ to export.',
    short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Checkpoint iteration (default: latest).',
    short_name='i',
)
flags.DEFINE_string(
    'output_dir', str(_REPO_ROOT / 'onnx_models'), 'Where to write the files.',
)
flags.DEFINE_float(
    'filter_cutoff_hz', 0.0,
    'First-order action filter cutoff [Hz] the policy was trained with. '
    'Only needed for checkpoints without environment.json (older runs); '
    'otherwise it is read from the run and this must match it.',
)


def load_agent_metadata(checkpoint_directory: pathlib.Path) -> dict:
    """Returns the agent metadata saved next to the checkpoints."""
    with open(checkpoint_directory / 'config.json') as f:
        return json.load(f)['agent_metadata']


def build_agent(metadata: dict) -> agent.Agent:
    """Rebuilds the trained agent's architecture from checkpoint metadata."""
    observation_size = {
        key: tuple(value) for key, value in metadata['observation_size'].items()
    }
    reference = {key: jnp.zeros(size) for key, size in observation_size.items()}

    def normalization(text: str, key: str) -> statistics.RunningStatistics:
        clip = re.search(r'clip=([0-9.]+)', text)
        return statistics.RunningStatistics(
            reference_input=reference[key],
            clip=float(clip.group(1)) if clip else None,
        )

    policy_key = metadata['policy_observation_key']
    value_key = metadata['value_observation_key']
    return agent.Agent(
        observation_size=observation_size,
        action_size=metadata['action_size'],
        state_dependent_std=False,
        policy_input_normalization=normalization(
            metadata['policy_input_normalization'], policy_key,
        ),
        value_input_normalization=normalization(
            metadata['value_input_normalization'], value_key,
        ),
        policy_layer_sizes=metadata['policy_layer_sizes'],
        value_layer_sizes=metadata['value_layer_sizes'],
        activation=_ACTIVATIONS[metadata['activation']],
        policy_observation_key=policy_key,
        value_observation_key=value_key,
    )


def restore_agent(
    model: agent.Agent, checkpoint_directory: pathlib.Path,
    iteration: int | None,
) -> None:
    """Loads checkpointed weights and statistics into `model` as float32."""
    manager = checkpoint_utilities.create_checkpoint_manager(
        checkpoint_directory=str(checkpoint_directory),
        max_to_keep=None,  # read-only: never prune the source run.
    )
    restored, _ = checkpoint_utilities.restore_training_state(
        manager=manager, agent=model, iteration=iteration,
    )
    state = jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, 'dtype') and x.dtype == jnp.float64 else x,
        restored.agent,
    )
    nnx.update(model, state)


def load_environment_settings(checkpoint_directory: pathlib.Path) -> dict:
    """Returns kp, kv, action_scale, control_timestep the run trained with.

    Uses `environment.json` saved by spot_train.py when present; older
    checkpoints fall back to the current EnvironmentConfig defaults.
    """
    path = checkpoint_directory / 'environment.json'
    defaults = spot_config.EnvironmentConfig()
    settings = {
        'kp': list(defaults.kp), 'kv': list(defaults.kv),
        'action_scale': (
            list(defaults.action_scale)
            if isinstance(defaults.action_scale, tuple) else defaults.action_scale
        ),
        'control_timestep': defaults.control_timestep,
        'command_range': [float(x) for x in spot_config.CommandConfig().command_range],
        'default_pose': _LEGACY_DEFAULT_POSE,
        'source': 'EnvironmentConfig defaults (no environment.json)',
    }
    if path.exists():
        with open(path) as f:
            settings.update(json.load(f))
        settings['source'] = 'environment.json'
    return settings


def resolve_filter_cutoff(flag_hz: float, recorded_hz: float) -> float:
    """Returns the filter cutoff [Hz], from the run's record and/or the flag."""
    if flag_hz and recorded_hz and not math.isclose(flag_hz, recorded_hz):
        raise ValueError(
            f'--filter_cutoff_hz={flag_hz} disagrees with the {recorded_hz} Hz '
            'recorded when the run was trained.'
        )
    return flag_hz or recorded_hz


def robot_constants(settings: dict) -> dict:
    """Joint names, default pose, action scale and gains (19-joint order)."""
    mj_model = mujoco.MjModel.from_xml_path(str(spot_config._DEFAULT_MJCF_PATH))
    n = policy_spec.NUM_LEG_JOINTS
    default_pose = np.asarray(settings['default_pose'], dtype=np.float64)
    scale = settings['action_scale']
    if scale is None:  # per-joint distance to the nearest joint limit
        low, high = mj_model.jnt_range[1:1 + n].T
        scale = np.minimum(high - default_pose[:n], default_pose[:n] - low)
    elif isinstance(scale, list):  # one value per joint type, tiled over the legs
        scale = np.tile(np.asarray(scale, dtype=np.float64), _NUM_LEGS)
    else:
        scale = np.full(n, scale)
    return {
        'joint_names': tuple(
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(1, mj_model.njnt)
        ),
        'default_pose': default_pose.astype(np.float32),
        'action_scale': np.asarray(scale, dtype=np.float32),
        'leg_lower': mj_model.jnt_range[1:1 + n, 0].astype(np.float32),
        'leg_upper': mj_model.jnt_range[1:1 + n, 1].astype(np.float32),
        'kp': np.concatenate([np.tile(settings['kp'], _NUM_LEGS), np.asarray(tc.ARM_KP)]),
        'kv': np.concatenate([np.tile(settings['kv'], _NUM_LEGS), np.asarray(tc.ARM_KV)]),
    }


def build_policy_step(
    model: agent.Agent, default_pose: np.ndarray, action_scale: np.ndarray,
    leg_lower: np.ndarray, leg_upper: np.ndarray, filter_alpha: float | None,
):
    """Returns f(sensors, state) -> (joint_position_targets, next_state).

    `filter_alpha` is None for policies trained without an action filter.
    """
    n = policy_spec.NUM_LEG_JOINTS
    default_pose = jnp.asarray(default_pose)
    action_scale = jnp.asarray(action_scale)
    leg_lower, leg_upper = jnp.asarray(leg_lower), jnp.asarray(leg_upper)
    sensor_splits = np.cumsum([f.size for f in policy_spec.SENSOR_FIELDS])[:-1]

    def policy_step(sensors: jax.Array, state: jax.Array):
        (linear, angular, gravity, command,
         joint_position, joint_velocity) = jnp.split(sensors, sensor_splits, axis=-1)
        previous_action = state[..., :n]
        previous_filtered = state[..., n:2 * n]

        # Observation Order:
        parts = [
            linear, angular, gravity, joint_position - default_pose,
            joint_velocity, previous_action, command,
        ]
        if filter_alpha is not None:
            parts.append(previous_filtered)
        observation = jnp.concatenate(parts, axis=-1)

        action, _ = model.get_actions(
            observation, jax.random.key(0), deterministic=True,
        )

        # State Update
        if filter_alpha is None:
            filtered, next_state = action, action
        else:
            filtered = filter_alpha * action + (1.0 - filter_alpha) * previous_filtered
            next_state = jnp.concatenate([action, filtered], axis=-1)

        # Training clips the leg setpoint to the joint limits (base.py).
        leg_targets = jnp.clip(
            default_pose[:n] + action_scale * filtered, leg_lower, leg_upper,
        )
        arm_targets = jnp.broadcast_to(
            default_pose[n:], leg_targets.shape[:-1] + (default_pose.shape[0] - n,),
        )
        return jnp.concatenate([leg_targets, arm_targets], axis=-1), next_state

    return policy_step


def rename_io(
    model_proto: onnx.ModelProto, input_names: list[str], output_names: list[str],
) -> onnx.ModelProto:
    """Names the graph's inputs and outputs in order (jax2onnx can't).

    A renamed tensor is rewritten everywhere it appears, including as an input
    of downstream nodes (an output tensor can also feed other nodes).
    """
    graph = model_proto.graph
    assert len(graph.input) == len(input_names), 'Unexpected number of inputs.'
    assert len(graph.output) == len(output_names), 'Unexpected number of outputs.'
    renames = {
        port.name: new
        for ports, names in ((graph.input, input_names), (graph.output, output_names))
        for port, new in zip(ports, names)
    }
    for node in graph.node:
        for field in (node.input, node.output):
            for i, name in enumerate(field):
                field[i] = renames.get(name, name)
    for port, new in zip(list(graph.input) + list(graph.output),
                         input_names + output_names):
        port.name = new
    return model_proto


def export_policy(
    policy_step, sensors_size: int, state_size: int, model_name: str,
) -> tuple[onnx.ModelProto, str]:
    """Traces the policy step to a simplified ONNX model and its export id."""
    proto = jax2onnx.to_onnx(
        fn=policy_step,
        inputs=[jnp.zeros((1, sensors_size), dtype=jnp.float32),
                jnp.zeros((1, state_size), dtype=jnp.float32)],
        opset=17,
        return_mode='proto',
        model_name=model_name,
        enable_double_precision=False,
    )
    proto = rename_io(
        proto, [policy_spec.SENSORS_NAME, policy_spec.STATE_NAME],
        [policy_spec.TARGETS_NAME, policy_spec.NEXT_STATE_NAME],
    )
    simplified, valid = simplify(proto)
    assert valid, 'Simplified ONNX model could not be validated.'
    # An id stored inside the model, so the driver can confirm at load that the
    # .onnx is the one its generated header describes (names and sizes alone
    # can't tell two exports of the same interface apart).
    export_id = hashlib.sha256(simplified.SerializeToString()).hexdigest()[:32]
    entry = simplified.metadata_props.add()
    entry.key, entry.value = 'export_id', export_id
    return simplified, export_id


def verify_layout(
    model: agent.Agent, session: onnxruntime.InferenceSession, robot: dict,
    filter_alpha: float | None,
) -> float:
    """Max |ONNX - training path| over random observations.

    Draws observations in the network's own layout, splits them into the
    sensors/state inputs, and compares the ONNX targets with the training
    path (policy, then filter, then default + scale * action). This checks
    the reordering the graph does, which a self-consistency check would miss.
    """
    n = policy_spec.NUM_LEG_JOINTS
    has_filter = filter_alpha is not None
    obs_size = (_OBS_SLICES['filter_state'].stop if has_filter
                else _STATE_SIZE_WITHOUT_FILTER)
    rng = np.random.default_rng(1)
    difference = 0.0
    for _ in range(_VERIFY_SAMPLES):
        obs = rng.normal(size=(1, obs_size)).astype(np.float32)
        obs[:, _OBS_SLICES['previous_action']] = rng.uniform(-1, 1, (1, n))
        if has_filter:
            obs[:, _OBS_SLICES['filter_state']] = rng.uniform(-1, 1, (1, n))
        sensors = np.concatenate([
            obs[:, _OBS_SLICES['base_linear_velocity']],
            obs[:, _OBS_SLICES['base_angular_velocity']],
            obs[:, _OBS_SLICES['projected_gravity']],
            obs[:, _OBS_SLICES['velocity_command']],
            obs[:, _OBS_SLICES['joint_position']] + robot['default_pose'],
            obs[:, _OBS_SLICES['joint_velocity']],
        ], axis=-1).astype(np.float32)
        state = obs[:, _OBS_SLICES['previous_action']]
        if has_filter:
            state = np.concatenate([state, obs[:, _OBS_SLICES['filter_state']]], axis=-1)

        action, _ = model.get_actions(
            jnp.asarray(obs), jax.random.key(0), deterministic=True,
        )
        action = np.asarray(action)
        if has_filter:
            action = (filter_alpha * action
                      + (1 - filter_alpha) * obs[:, _OBS_SLICES['filter_state']])
        expected_legs = np.clip(
            robot['default_pose'][:n] + robot['action_scale'] * action,
            robot['leg_lower'], robot['leg_upper'],
        )
        targets, _ = session.run(None, {
            policy_spec.SENSORS_NAME: sensors, policy_spec.STATE_NAME: state,
        })
        difference = max(
            difference,
            float(np.max(np.abs(targets[:, :n] - expected_legs))),
            float(np.max(np.abs(targets[:, n:] - robot['default_pose'][n:]))),
        )
    assert difference < _VERIFY_TOLERANCE, f'Layout mismatch: {difference}.'
    return difference


def verify_recurrence(
    policy_step, session: onnxruntime.InferenceSession, sensors_size: int,
    state_size: int,
) -> float:
    """Max |ONNX - JAX| over a rollout that feeds next_state back as state."""
    rng = np.random.default_rng(2)
    state_onnx = np.zeros((1, state_size), np.float32)
    state_jax = jnp.zeros((1, state_size), jnp.float32)
    difference = 0.0
    for _ in range(_VERIFY_STEPS):
        sensors = rng.normal(size=(1, sensors_size)).astype(np.float32)
        targets, state_onnx = session.run(None, {
            policy_spec.SENSORS_NAME: sensors, policy_spec.STATE_NAME: state_onnx,
        })
        expected, state_jax = policy_step(jnp.asarray(sensors), state_jax)
        difference = max(
            difference,
            float(np.max(np.abs(targets - np.asarray(expected)))),
            float(np.max(np.abs(state_onnx - np.asarray(state_jax)))),
        )
    assert difference < _VERIFY_TOLERANCE, f'Recurrence mismatch: {difference}.'
    return difference


def write_test_vectors(
    policy_step, sensors_size: int, state_size: int, path: pathlib.Path,
    default_pose: np.ndarray,
) -> None:
    """Writes reference (known-answer) vectors from the JAX policy for C++ tests.

    A C++ test runs the real .onnx through its driver on these inputs and
    compares with the expected outputs, which checks the runtime, the ops and
    the driver end to end, independent of the Python export.
    """
    rng = np.random.default_rng(3)

    def run(sensors, state):
        targets, next_state = policy_step(jnp.asarray(sensors[None]), jnp.asarray(state[None]))
        return np.asarray(targets)[0].tolist(), np.asarray(next_state)[0].tolist()

    level = np.zeros(sensors_size, np.float32)  # standing still at the default pose
    level[6:9] = (0.0, 0.0, -1.0)
    level[12:12 + policy_spec.NUM_JOINTS] = default_pose
    cases = [level] + [
        rng.normal(size=sensors_size).astype(np.float32) for _ in range(4)
    ]
    single = []
    for sensors in cases:
        state = rng.uniform(-1, 1, state_size).astype(np.float32)
        if sensors is cases[0]:
            state = np.zeros(state_size, np.float32)
        targets, next_state = run(sensors, state)
        single.append({'sensors': sensors.tolist(), 'state': state.tolist(),
                       'joint_position_targets': targets, 'next_state': next_state})

    state = np.zeros(state_size, np.float32)
    steps = []
    for _ in range(20):
        sensors = rng.normal(size=sensors_size).astype(np.float32)
        targets, next_state = run(sensors, state)
        steps.append({'sensors': sensors.tolist(), 'joint_position_targets': targets,
                      'next_state': next_state})
        state = np.asarray(next_state, np.float32)
    with open(path, 'w') as f:
        json.dump({
            'description': 'Reference vectors from the JAX policy. single_step: '
                           'independent (sensors, state) -> (targets, next_state). '
                           'rollout: start from zero state and feed next_state back.',
            'tolerance': _VERIFY_TOLERANCE, 'single_step': single,
            'rollout': {'initial_state': [0.0] * state_size, 'steps': steps},
        }, f)


def main(argv=None):
    name = FLAGS.checkpoint_name
    checkpoint_directory = _REPO_ROOT / 'checkpoints' / name
    metadata = load_agent_metadata(checkpoint_directory)
    obs_size = metadata['observation_size'][metadata['policy_observation_key']][0]

    settings = load_environment_settings(checkpoint_directory)
    cutoff_hz = resolve_filter_cutoff(
        FLAGS.filter_cutoff_hz, settings.get('filter_cutoff_hz', 0.0),
    )
    has_filter = obs_size != _STATE_SIZE_WITHOUT_FILTER
    if has_filter and not cutoff_hz:
        raise ValueError(
            f'Observation size {obs_size} includes filter state, but the '
            'checkpoint has no environment.json; pass --filter_cutoff_hz '
            'with the cutoff the policy was trained with.'
        )
    if not has_filter and cutoff_hz:
        raise ValueError(
            f'Observation size {obs_size} has no filter state, but a filter '
            f'cutoff of {cutoff_hz} Hz was given or recorded.'
        )

    dt = settings['control_timestep']
    alpha = None
    if has_filter:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        alpha = dt / (tau + dt)

    model = build_agent(metadata)
    restore_agent(model, checkpoint_directory, FLAGS.checkpoint_iteration)
    robot = robot_constants(settings)
    policy_step = build_policy_step(
        model, robot['default_pose'], robot['action_scale'],
        robot['leg_lower'], robot['leg_upper'], alpha,
    )
    sensors_size = policy_spec.total_size(policy_spec.SENSOR_FIELDS)
    state_size = policy_spec.total_size(policy_spec.state_fields(has_filter))

    proto, export_id = export_policy(policy_step, sensors_size, state_size, f'{name}_spot')
    session = onnxruntime.InferenceSession(
        proto.SerializeToString(), providers=['CPUExecutionProvider'],
    )
    layout_error = verify_layout(model, session, robot, alpha)
    recurrence_error = verify_recurrence(policy_step, session, sensors_size, state_size)

    output_dir = pathlib.Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f'{name}_spot.onnx'
    onnx.save_model(proto, str(onnx_path))
    spec = policy_spec.PolicySpec(
        checkpoint=name,
        iteration=(FLAGS.checkpoint_iteration
                   if FLAGS.checkpoint_iteration is not None else 'latest'),
        onnx_file=onnx_path.name,
        onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        export_id=export_id,
        control_timestep_s=dt,
        command_range=tuple(round(float(x), 6) for x in settings['command_range']),
        has_filter=has_filter,
        filter_cutoff_hz=cutoff_hz,
        filter_alpha=alpha or 0.0,
        joint_names=robot['joint_names'],
        default_pose=tuple(float(x) for x in robot['default_pose']),
        action_scale=tuple(float(x) for x in robot['action_scale']),
        leg_lower=tuple(float(x) for x in robot['leg_lower']),
        leg_upper=tuple(float(x) for x in robot['leg_upper']),
        kp=tuple(float(x) for x in robot['kp']),
        kv=tuple(float(x) for x in robot['kv']),
        settings_source=settings['source'],
    )
    spec.write_json(output_dir / f'{name}_spot.json')
    spec.write_cpp_header(output_dir / f'{name}_spot_spec.h')
    write_test_vectors(
        policy_step, sensors_size, state_size,
        output_dir / f'{name}_spot_test_vectors.json', robot['default_pose'],
    )

    print(f'Wrote {onnx_path} plus .json, _spec.h and _test_vectors.json')
    print(f'  layout error {layout_error:.2e}, {_VERIFY_STEPS}-step recurrence '
          f'error {recurrence_error:.2e}, settings from {settings["source"]}')


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_name')
    app.run(main)
