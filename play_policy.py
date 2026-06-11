from absl import app, flags, logging

import os
import functools
from pathlib import Path
import pickle
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np
import numpy.typing as npt

import flax.nnx as nnx

import mujoco
import mujoco.viewer

from mujoco import mjx

from training.envs.unitree_go2 import unitree_go2_joystick
from training.envs.unitree_go2 import config
import training.envs.utilities.filter as filters

import training.statistics as statistics
import training.algorithms.ppo.agent as agent

from training import checkpoint_utilities

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

logging.set_verbosity(logging.FATAL)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'policy_checkpoint_path', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_string(
    'parameter_checkpoint_path', None, 'Parameter checkpoint path to load.', short_name='p',
)


def main(argv=None):
    # Rehydrate Model from Parameter Checkpoint:
    model_params = None
    if FLAGS.parameter_checkpoint_path is not None:
        parameter_checkpoint_path = Path(FLAGS.parameter_checkpoint_path) / 'regressed_params.pkl'
        with open(parameter_checkpoint_path, 'rb') as f:
            params = pickle.load(f)

        # Get Regressed Parameters:
        model_params = {
            k: v
            for k, v in params.items()
            if not k.startswith('initial_')
        }

    # Configure Environment:
    control_timestep = 0.02

    # Use Motor Model:
    motor_config = config.MotorConfig()

    # First Order Filter:
    cutoff_frequency = 4.0
    tau = 1 / (2 * jnp.pi * cutoff_frequency)
    alpha = control_timestep / (tau + control_timestep)
    filter_impl = filters.FirstOrderFilter(
        action_dim=12,
        alpha=alpha,
    )
    filter_state = filter_impl.init()

    scene = 'scene_mjx_vendor_torque_rough.xml'
    environment_config = config.EnvironmentConfig(
        filename=scene,
        action_scale=None,
        control_timestep=0.02,
        optimizer_timestep=0.004,
    )

    # Initialize Environment and Simulation:
    env = unitree_go2_joystick.UnitreeGo2Env(
        environment_config=environment_config,
        motor_config=motor_config,
        model_params=model_params,
        filter_impl=filter_impl,
    )

    data = mujoco.MjData(env._mj_model)
    mujoco.mj_resetDataKeyframe(env._mj_model, data, 0)
    control_rate = 0.02
    n_substeps = int(control_rate / env._mj_model.opt.timestep)

    # Set Motor Model:
    if env.motor_config is not None:
        def motor_model(mj_data: mujoco.MjData, target_qpos: npt.NDArray) -> npt.NDArray:
            # Extract Joint States:
            joint_positions = mj_data.qpos[7:]
            joint_velocities = mj_data.qvel[6:]

            # PD Control Law
            desired_torque = env.motor_config.kp * (target_qpos - joint_positions) \
                - env.motor_config.kv * joint_velocities

            # Torque Speed Curve:
            available_torque = env.motor_config.tau_max - (env.motor_config.damping_slope * np.abs(joint_velocities))
            available_torque = np.maximum(available_torque, 0.0)

            # Apply Torque Limits:
            torque = np.clip(desired_torque, -available_torque, available_torque)

            return torque
    else:
        def motor_model(mj_data: mujoco.MjData, target_qpos: npt.NDArray) -> npt.NDArray:
            return target_qpos
        
    # Utility Functions:
    def simulation_step(env: unitree_go2_joystick.UnitreeGo2Env, data: mujoco.MjData, action: npt.NDArray, n_substeps: int) -> mujoco.MjData:
        # Compute Target Joint Positions from Action:
        target_qpos = env.default_pose + action * env.action_scale
        target_qpos = np.clip(target_qpos, env.joint_lb, env.joint_ub)

        # Run Physics Substeps:
        for _ in range(n_substeps):
            ctrl = motor_model(data, target_qpos)
            data.ctrl = ctrl
            mujoco.mj_step(env._mj_model, data)

        return data

    # Set Simulation Step Function:
    step_fn = functools.partial(
        simulation_step,
        env=env,
        n_substeps=n_substeps,
    )

    # Setup Agent and Rehydrate Policy from Checkpoint:
    observation_size = env.observation_size
    action_size = env.action_size
    reference_observation = {
        key: jnp.zeros(value) for key, value in observation_size.items()
    }

    # Setup agent:
    policy_layer_size = [512, 256, 128,]
    value_layer_size = [512, 256, 128,]
    activation_fn = jax.nn.swish
    hidden_init = jax.nn.initializers.orthogonal(jnp.sqrt(2.0))
    output_init = jax.nn.initializers.orthogonal(0.01)
    policy_kernel_init = [hidden_init] * len(policy_layer_size) + [output_init]
    value_kernel_init = [hidden_init] * len(value_layer_size) + [output_init]
    policy_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["state"],
    )
    value_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["privileged_state"],
    )
    model = agent.Agent(
        observation_size=observation_size,
        action_size=action_size,
        state_dependent_std=False,
        policy_input_normalization=policy_input_normalization,
        value_input_normalization=value_input_normalization,
        policy_layer_sizes=policy_layer_size,
        value_layer_sizes=value_layer_size,
        activation=activation_fn,
        policy_kernel_init=policy_kernel_init,
        value_kernel_init=value_kernel_init,
        policy_observation_key="state",
        value_observation_key="privileged_state",
    )

    if FLAGS.policy_checkpoint_path is not None:
        restore_directory = Path(__file__).parent / FLAGS.policy_checkpoint_path
        restore_manager = checkpoint_utilities.create_checkpoint_manager(
            checkpoint_directory=restore_directory,
        )
        restored_checkpoint, _ = checkpoint_utilities.restore_training_state(
            manager=restore_manager,
            agent=model,
        )

        nnx.update(model, restored_checkpoint.agent)

    # Create Inference Wrapper:
    def inference_wrapper(observation: jax.Array) -> jax.Array:
        dummy_key = jax.random.key(0)

        actions, info = model.get_actions(
            observation, 
            dummy_key, 
            deterministic=True
        )

        return actions

    inference_fn = jax.jit(inference_wrapper)
    observation_fn = jax.jit(env.get_observation)


    # Initialize Observation History:
    observation = reference_observation
    action = np.zeros_like(env.default_ctrl)
    command = np.array([0.0, 0.0, 0.0])

    key = jax.random.key(0)


    ## keyboard inputs instead of joystick commands
    
    # Keyboard Command State (shared between viewer key_callback and sim loop):
    command_state = {'forward': 0.0, 'lateral': 0.0, 'rotation': 0.0}
    termination_flag = [False]
    command_increment = 0.25

    def key_callback(keycode):
        try:
            k = chr(keycode)
        except ValueError:
            k = ''
        if k == 'W':
            command_state['forward'] = min(1.0, command_state['forward'] + command_increment)
        elif k == 'S':
            command_state['forward'] = max(-1.0, command_state['forward'] - command_increment)
        elif k == 'A':
            command_state['lateral'] = min(1.0, command_state['lateral'] + command_increment)
        elif k == 'D':
            command_state['lateral'] = max(-1.0, command_state['lateral'] - command_increment)
        elif k == 'Q':
            command_state['rotation'] = min(1.0, command_state['rotation'] + command_increment)
        elif k == 'E':
            command_state['rotation'] = max(-1.0, command_state['rotation'] - command_increment)
        elif k == ' ':
            command_state['forward'] = 0.0
            command_state['lateral'] = 0.0
            command_state['rotation'] = 0.0
        elif keycode == 256:
            termination_flag[0] = True
        print(
            f"command: fwd={command_state['forward']:+.2f} "
            f"lat={command_state['lateral']:+.2f} "
            f"rot={command_state['rotation']:+.2f}"
        )

    with mujoco.viewer.launch_passive(env._mj_model, data, key_callback=key_callback) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5

        print("Keyboard controls: W/S fwd/back, A/D lateral, Q/E yaw, Space zero, ESC quit")

        while viewer.is_running() and not termination_flag[0]:
            # Walking Policy:
            x_scale, y_scale, z_scale = 1.5, 1.0, 3.0
            command = np.array([
                x_scale * command_state['forward'],
                y_scale * command_state['lateral'],
                z_scale * command_state['rotation'],
            ])
            command = np.where(np.abs(command) < 0.1, 0.0, command)

            step_time = time.time()

            # Get Observation:
            state_info = {
                # Required For Policy Inference:
                'previous_action': action,
                'filter_state': filter_state,
                'command': command,
                'rng': key,
                # Dummy values for Critic Network (Not Used for Inference):
                'previous_contact': np.zeros(4),
                'feet_air_time': np.zeros(4),
                'feet_contact_time': np.zeros(4),
                'previous_air_time': np.zeros(4),
                'previous_contact_time': np.zeros(4),
                'swing_peak': np.zeros(4),
                'steps_until_next_disturbance': 0,
                'steps_since_previous_disturbance': 0,
            }

            mjx_data = mjx.put_data(env._mj_model, data)
            observation = observation_fn(
                mjx_data,
                state_info,
            )

            # Inference Policy for Action:
            action = inference_fn(observation)
            action, filter_state = jax.jit(filter_impl.apply)(action, filter_state)

            # Move from GPU to CPU: This should be a blocking operation
            action = np.asarray(action)
            
            # Step Simulation:
            data = step_fn(data=data, action=action)

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == '__main__':
    app.run(main)
