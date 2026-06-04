import os
from absl import flags, logging
import jax
import mujoco
import numpy as np
import numpy.typing as npt

from training.envs.unitree_go2 import unitree_go2_joystick
import training.statistics as statistics
import training.algorithms.ppo.agent as agent
import jax.numpy as jnp
import cv2
import time


def set_env_flags():
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

    return FLAGS



# Set Motor Model:
def set_motor_model(env):
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

    return motor_model


# Utility Functions:
def simulation_step(env: unitree_go2_joystick.UnitreeGo2Env, motor_model, data: mujoco.MjData, action: npt.NDArray, n_substeps: int) -> mujoco.MjData:
    # Compute Target Joint Positions from Action:
    target_qpos = env.default_pose + action * env.action_scale
    target_qpos = np.clip(target_qpos, env.joint_lb, env.joint_ub)

    # Run Physics Substeps:
    for _ in range(n_substeps):
        ctrl = motor_model(data, target_qpos)
        data.ctrl = ctrl
        mujoco.mj_step(env._mj_model, data)

    return data

# Agent model setup
def setup_agent(env: unitree_go2_joystick.UnitreeGo2Env):
    
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

    return model, reference_observation


# Live depth visualization using openCV.
def live_depth(depth_array: np.ndarray, max_visual_depth: float, control_rate: float, step_time: float) -> None:
    depth_normalized = np.clip(depth_array, 0, max_visual_depth) / max_visual_depth
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    cv2.imshow("ZED - Depth", depth_colored)

    sleep_time = control_rate - (time.time() - step_time)
    if sleep_time > 0:
        delay_ms = max(1, int(sleep_time * 1000))
        cv2.waitKey(delay_ms)
    else:
        cv2.waitKey(1)

# Utility function to close windows after loop ends.
def destroy_depth_windows() -> None:
    cv2.destroyAllWindows()

