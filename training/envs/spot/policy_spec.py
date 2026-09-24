"""Description of an exported Spot policy: tensor layouts, gains, joints.

`to_onnx.py` builds a `PolicySpec` and writes it two ways from the same data:
a JSON file (human/tool readable) and a C++ header of constexpr constants, so
the drivers can size buffers and index tensors at compile time. The only
runtime check left is comparing the loaded model's tensor names, shapes and
hash to the header once at load.

The ONNX interface is a stateless graph with explicit recurrent state:
    inputs : sensors [1, 50], state [1, 12 or 24]
    outputs: joint_position_targets [1, 19], next_state [1, 12 or 24]
The caller feeds `next_state` back as `state` on the following tick, starting
from zeros.
"""

import dataclasses
import json
import pathlib

NUM_LEG_JOINTS = 12
NUM_ARM_JOINTS = 7
NUM_JOINTS = NUM_LEG_JOINTS + NUM_ARM_JOINTS

SENSORS_NAME = 'sensors'
STATE_NAME = 'state'
TARGETS_NAME = 'joint_position_targets'
NEXT_STATE_NAME = 'next_state'


@dataclasses.dataclass(frozen=True)
class Field:
    """A named slice of a tensor."""
    name: str
    size: int
    unit: str = ''
    note: str = ''


# Sensor order matches spot-drivers' PolicyBaseDriver::make_observation().
SENSOR_FIELDS = (
    Field('base_linear_velocity', 3, 'm/s', 'body frame, at the body origin'),
    Field('base_angular_velocity', 3, 'rad/s', 'body frame'),
    Field('projected_gravity', 3, '', 'body frame unit vector, (0,0,-1) when level'),
    Field('velocity_command', 3, 'm/s, m/s, rad/s', 'vx, vy, yaw rate'),
    Field('joint_position', NUM_JOINTS, 'rad', 'absolute; 12 legs then 7 arm'),
    Field('joint_velocity', NUM_JOINTS, 'rad/s', '12 legs then 7 arm'),
)
TARGET_FIELDS = (
    Field('leg_position_target', NUM_LEG_JOINTS, 'rad', 'from the policy'),
    Field('arm_position_target', NUM_ARM_JOINTS, 'rad',
          'the arm default pose the policy was trained with'),
)


def state_fields(has_filter: bool) -> tuple[Field, ...]:
    """Recurrent state layout: the raw previous action, then the filter state."""
    fields = (Field('previous_action', NUM_LEG_JOINTS, '',
                    'raw policy output, before any filter'),)
    if has_filter:
        fields += (Field('filter_state', NUM_LEG_JOINTS, '',
                         'last filtered action'),)
    return fields


def total_size(fields: tuple[Field, ...]) -> int:
    return sum(field.size for field in fields)


def with_offsets(fields: tuple[Field, ...]) -> list[dict]:
    """Returns each field as a dict with its offset into the tensor."""
    result, offset = [], 0
    for field in fields:
        result.append({**dataclasses.asdict(field), 'offset': offset})
        offset += field.size
    return result


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    """Everything the deployment side needs to know about one exported policy."""
    checkpoint: str
    iteration: int | str
    onnx_file: str
    onnx_sha256: str
    export_id: str
    control_timestep_s: float
    command_range: tuple[float, float, float]
    has_filter: bool
    filter_cutoff_hz: float
    filter_alpha: float
    joint_names: tuple[str, ...]
    default_pose: tuple[float, ...]
    action_scale: tuple[float, ...]
    leg_lower: tuple[float, ...]
    leg_upper: tuple[float, ...]
    kp: tuple[float, ...]
    kv: tuple[float, ...]
    settings_source: str

    @property
    def sensors_size(self) -> int:
        return total_size(SENSOR_FIELDS)

    @property
    def state_size(self) -> int:
        return total_size(state_fields(self.has_filter))

    def to_dict(self) -> dict:
        """The JSON description."""
        return {
            'model': {'file': self.onnx_file, 'sha256': self.onnx_sha256,
                      'export_id': self.export_id, 'opset': 17},
            'policy': {
                'checkpoint': self.checkpoint, 'iteration': self.iteration,
                'control_timestep_s': self.control_timestep_s,
                'control_rate_hz': round(1.0 / self.control_timestep_s, 6),
                'command_range': {
                    'vx': self.command_range[0], 'vy': self.command_range[1],
                    'yaw_rate': self.command_range[2],
                    'note': 'velocity commands were sampled uniformly in '
                            '[-range, +range]; stay inside it',
                },
                'settings_source': self.settings_source,
            },
            'inputs': [
                {'name': SENSORS_NAME, 'shape': [1, self.sensors_size],
                 'fields': with_offsets(SENSOR_FIELDS)},
                {'name': STATE_NAME, 'shape': [1, self.state_size],
                 'initial_value': 'zeros', 'fed_from_output': NEXT_STATE_NAME,
                 'fields': with_offsets(state_fields(self.has_filter))},
            ],
            'outputs': [
                {'name': TARGETS_NAME, 'shape': [1, NUM_JOINTS],
                 'fields': with_offsets(TARGET_FIELDS)},
                {'name': NEXT_STATE_NAME, 'shape': [1, self.state_size],
                 'feeds_input': STATE_NAME,
                 'fields': with_offsets(state_fields(self.has_filter))},
            ],
            'joints': {'names': list(self.joint_names),
                       'default_pose': list(self.default_pose)},
            'pd_gains': {'kp': list(self.kp), 'kv': list(self.kv),
                         'note': 'position command with zero velocity setpoint: '
                                 'torque = kp*(q_target - q) - kv*qdot'},
            'baked_into_graph': {
                'default_pose_subtraction': True,
                'action_scale': list(self.action_scale),
                'leg_position_limits': {
                    'lower': list(self.leg_lower), 'upper': list(self.leg_upper),
                    'note': 'leg targets are clipped to these, as in training',
                },
                'action_filter': None if not self.has_filter else {
                    'type': 'first_order', 'cutoff_hz': self.filter_cutoff_hz,
                    'alpha': self.filter_alpha,
                    'equation': 'y[n] = alpha*x[n] + (1-alpha)*y[n-1]',
                },
            },
        }

    def write_json(self, path: pathlib.Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_cpp_header(self) -> str:
        """A constexpr header in spot-drivers' snake_case style."""
        def literal(value: float) -> str:
            text = f'{value:.7g}'
            if not any(c in text for c in '.eE'):
                text += '.0'  # `45f` is not a valid C++ literal
            return text + 'f'

        def array(name: str, values, ctype: str = 'float') -> str:
            body = ', '.join(literal(v) for v in values)
            return (f'inline constexpr std::array<{ctype}, {len(values)}> '
                    f'{name} = {{{body}}};')

        def offsets(prefix: str, fields: tuple[Field, ...]) -> list[str]:
            lines, offset = [], 0
            for field in fields:
                lines.append(f'inline constexpr std::size_t {prefix}_{field.name}_offset = {offset};')
                lines.append(f'inline constexpr std::size_t {prefix}_{field.name}_size = {field.size};')
                offset += field.size
            return lines

        state = state_fields(self.has_filter)
        lines = [
            f'// Generated by training/envs/spot/to_onnx.py from checkpoint '
            f'"{self.checkpoint}". Do not edit.',
            '#pragma once', '',
            '#include <array>', '#include <cstddef>', '#include <string_view>', '',
            'namespace spot_drivers::driver::policy_spec {', '',
            f'inline constexpr std::string_view onnx_file = "{self.onnx_file}";',
            f'inline constexpr std::string_view onnx_sha256 = "{self.onnx_sha256}";',
            '// Also stored inside the .onnx as metadata key "export_id"; compare at load.',
            f'inline constexpr std::string_view export_id = "{self.export_id}";',
            f'inline constexpr double control_period_s = {self.control_timestep_s};',
            f'inline constexpr double control_rate_hz = {1.0 / self.control_timestep_s:.9g};',
            'inline constexpr std::array<float, 3> command_range = '
            f'{{{literal(self.command_range[0])}, {literal(self.command_range[1])}, '
            f'{literal(self.command_range[2])}}};  // |vx|, |vy|, |yaw rate| limits',
            f'inline constexpr bool has_action_filter = '
            f'{"true" if self.has_filter else "false"};', '',
            f'inline constexpr std::size_t num_leg_joints = {NUM_LEG_JOINTS};',
            f'inline constexpr std::size_t num_arm_joints = {NUM_ARM_JOINTS};',
            'inline constexpr std::size_t num_joints = num_leg_joints + num_arm_joints;', '',
            '// Tensors (names and sizes are checked once against the model at load).',
            f'inline constexpr std::string_view sensors_name = "{SENSORS_NAME}";',
            f'inline constexpr std::size_t sensors_size = {self.sensors_size};',
            f'inline constexpr std::string_view state_name = "{STATE_NAME}";',
            f'inline constexpr std::size_t state_size = {self.state_size};',
            f'inline constexpr std::string_view joint_position_targets_name = "{TARGETS_NAME}";',
            'inline constexpr std::size_t joint_position_targets_size = num_joints;',
            f'inline constexpr std::string_view next_state_name = "{NEXT_STATE_NAME}";',
            'inline constexpr std::size_t next_state_size = state_size;', '',
            '// All tensors are float32 with shape [1, size].',
            'inline constexpr std::array<std::string_view, 2> input_names = {sensors_name, state_name};',
            'inline constexpr std::array<std::size_t, 2> input_sizes = {sensors_size, state_size};',
            'inline constexpr std::array<std::string_view, 2> output_names = '
            '{joint_position_targets_name, next_state_name};',
            'inline constexpr std::array<std::size_t, 2> output_sizes = '
            '{joint_position_targets_size, next_state_size};', '',
            '// Offsets into the sensors input and the state input/next_state output.',
            *offsets('sensors', SENSOR_FIELDS), '',
            *offsets('state', state), '',
            '// Joint order: 12 legs (FL, FR, RL, RR; hip, thigh, calf) then 7 arm.',
            array('policy_default_pose', self.default_pose),
            array('policy_kp', self.kp), array('policy_kv', self.kv),
            '',
            '}  // namespace spot_drivers::driver::policy_spec', '',
        ]
        return '\n'.join(lines)

    def write_cpp_header(self, path: pathlib.Path) -> None:
        path.write_text(self.to_cpp_header())
