from absl import app, flags
import os
import functools
import time

import pygame

import jax
import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer

from training.envs.unitree_go2 import unitree_go2_joystick as unitree_go2
from training.algorithms.ppo.load_utilities import load_policy


jax.config.update("jax_enable_x64", True)
pygame.init()

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
    filename = 'scene_mjx.xml'
    action_scale = 0.5
    time_window = 5

    env = unitree_go2.UnitreeGo2Env(
        filename=filename,
        action_scale=action_scale,
        time_window=time_window,
    )

    model_path = os.path.join(
        os.path.dirname(__file__),
        f'{env.filename}',
    )
    model = mujoco.MjModel.from_xml_path(
        model_path,
    )
    model.opt.timestep = 0.004

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    control_rate = 0.02
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

    # Setup Joystick:
    joysticks = {}

    key = jax.random.key(0)
    termination_flag = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        while viewer.is_running() and not termination_flag:
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    joy = pygame.joystick.Joystick(event.device_index)
                    joysticks[joy.get_instance_id()] = joy
                    print(f"Joystick {joy.get_instance_id()} connencted")

                if event.type == pygame.JOYDEVICEREMOVED:
                    del joysticks[event.instance_id]
                    print(f"Joystick {event.instance_id} disconnected")

            for joystick in joysticks.values():
                if joystick.get_button(6) == 1:
                    termination_flag = True

                # XBox One:
                forward_command = -1 * joystick.get_axis(1)
                lateral_command = -1 * joystick.get_axis(0)
                rotation_command = -1 * joystick.get_axis(3)
                alpha = (joystick.get_axis(5) + 1) / 2

            # Walking Policy:
            command = np.array([
                forward_command, lateral_command, rotation_command,
            ])
            command = np.where(np.abs(command) < 0.1, 0.0, command)
            command = np.clip(command, -1.0, 1.0)

            step_time = time.time()
            action_rng, key = jax.random.split(key)

            # Get Observation:
            observation = env.np_observation(
                mj_data=data,
                command=command,
                observation_history=observation['state'],
                previous_action=action,
                add_noise=False,
            )
            action, _ = jax.block_until_ready(
                inference_fn(observation, action_rng),
            )

            ctrl = controller_fn(action)
            ctrl = (1 - alpha) * env.default_ctrl + (alpha) * ctrl
            data.ctrl = ctrl

            for _ in range(num_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == '__main__':
    app.run(main)
