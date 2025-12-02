from absl import app, flags
import os
import functools

import dataclasses
import copy
import csv
from pathlib import Path

import jax
import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt

from training.envs.unitree_go2 import unitree_go2_joystick as unitree_go2
from training.envs.unitree_go2 import config

from training.algorithms.ppo.load_utilities import load_policy


jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


def controller(
    action: npt.ArrayLike,
    default_control: npt.ArrayLike,
    action_scale: float,
) -> np.ndarray:
    motor_targets = default_control + action * action_scale
    return motor_targets


def main(argv=None):
    filename = "scene_vendored_mjx.xml"
    env_config = config.EnvironmentConfig(
        filename=filename,
        action_scale=0.5,
        control_timestep=0.02,
        optimizer_timestep=0.004,
    )

    env = unitree_go2.UnitreeGo2Env(
        env_config=env_config,
    )

    model_path = os.path.join(
        os.path.dirname(__file__),
        f'{env.filepath}',
    )

    model = mujoco.MjModel.from_xml_path(
        model_path,
    )

    model.opt.timestep = env_config.optimizer_timestep

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    control_rate = env_config.control_timestep
    num_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params, _ = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
        restore_iteration=FLAGS.checkpoint_iteration,
    )
    inference_function = make_policy(params, deterministic=True)
    inference_fn = jax.jit(inference_function)

    # Controller:
    controller_fn = functools.partial(
        controller,
        default_control=env.default_ctrl,
        action_scale=env.action_scale,
    )

    # Initialize Observation History:
    observation = {
        'state': np.zeros(env.num_observations),
        'privileged_state': np.zeros(env.num_privileged_observations),
    }
    action = np.zeros_like(env.default_ctrl)
    command = np.array([0.0, 0.0, 0.0])

    key = jax.random.key(0)
    termination_flag = False

    @dataclasses.dataclass
    class SimulationData:
        time: float
        ctrl: npt.ArrayLike
        qpos: npt.ArrayLike
        qvel: npt.ArrayLike
        actuator_force: npt.ArrayLike
    
    def record_data(data: mujoco.MjData) -> SimulationData:
        return SimulationData(
            time=copy.deepcopy(data.time),
            ctrl=copy.deepcopy(data.ctrl),
            qpos=copy.deepcopy(data.qpos),
            qvel=copy.deepcopy(data.qvel),
            actuator_force=copy.deepcopy(data.actuator_force),
        )

    data_history = [record_data(data)]

    while not termination_flag:
        # Walking Policy:
        command = np.array([1.0, 0.0, 0.0])

        action_rng, key = jax.random.split(key)

        # Get Observation:
        observation = env.np_observation(
            mj_data=data,
            command=command,
            previous_action=action,
            add_noise=False,
        )
        action, _ = jax.block_until_ready(
            inference_fn(observation, action_rng),
        )

        ctrl = controller_fn(action)
        data.ctrl = ctrl

        for _ in range(num_steps):
            mujoco.mj_step(model, data)  # type: ignore
            data_history.append(record_data(data))

        if data.time >= 5.0:
            termination_flag = True

    # Parse Data History and write to csv:
    output_filename = "simulation_history.csv"
    output_path = Path(output_filename)
    with output_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for data in data_history:
            timestep = data.time
            command = data.ctrl
            positions = data.qpos[7:]
            velocities = data.qvel[6:]
            torques = data.actuator_force
            row_data = (
                [timestep] +
                list(positions) + 
                list(velocities) + 
                list(torques) +
                list(command)
            )
            writer.writerow(row_data)


if __name__ == '__main__':
    app.run(main)
