from absl import app, flags, logging

import os
import functools
from pathlib import Path
import pickle
import time

import jax

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
from play_utils import utils

import cv2

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

    motor_model = utils.set_motor_model(env)

    # Set Simulation Step Function:
    step_fn = functools.partial(
        utils.simulation_step,
        env=env,
        motor_model=motor_model,
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
    
    renderer = mujoco.Renderer(env._mj_model, height=240, width=424)
    renderer.enable_depth_rendering()

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

            # Get Observation (CPU path — no MJX round-trip, so heightfield
            # contact counts can't overflow MJX's fixed contact buffer):
            obs = env.np_observation(
                data,
                command=command,
                previous_action=action,
                add_noise=False,
            )
            # np_observation predates the action filter, so it omits the filter
            # term. Append the last filtered action to match the trained layout
            # ([..., command, filter_observation]).
            filter_observation = np.asarray(filter_state.last_filtered_action)
            state = np.concatenate([obs['state'], filter_observation])
            observation = {
                'state': state,
                'privileged_state': obs['privileged_state'],
            }

            # Inference Policy for Action:
            action = inference_fn(observation)
            action, filter_state = jax.jit(filter_impl.apply)(action, filter_state)

            # Move from GPU to CPU: This should be a blocking operation
            action = np.asarray(action)

            # Step Simulation:
            data = step_fn(data=data, action=action)

            viewer.sync()

            renderer.update_scene(data, camera="zedm") 
            depth_array = renderer.render()

            max_visual_depth = 5.0
            
            depth_array = np.nan_to_num(
                depth_array, 
                nan=max_visual_depth, 
                posinf=max_visual_depth, 
                neginf=0.0
            )

            depth_normalized = np.clip(depth_array, 0, max_visual_depth) / max_visual_depth
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)

            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
            cv2.imshow("ZED - Depth", depth_colored)

            # Manage Control Rate and OpenCV Event Loop
            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                delay_ms = max(1, int(sleep_time * 1000))
                cv2.waitKey(delay_ms)
            else:
                cv2.waitKey(1)

    # Cleanup OpenCV windows when the viewer closes
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(main)

 