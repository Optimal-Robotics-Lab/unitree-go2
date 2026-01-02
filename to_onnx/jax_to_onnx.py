import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)
sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags

import functools

import jax2onnx

from training.envs.unitree_go2.unitree_go2_joystick import UnitreeGo2Env
from training.checkpoint_utilities import restore_checkpoint

from plugins import log1p


os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)


def main(argv=None):
    if not FLAGS.checkpoint_name:
        print("Error: Please provide a checkpoint name using -c or --checkpoint_name")
        return

    print(f"Loading JAX policy from checkpoint: {FLAGS.checkpoint_name}...")
    env = UnitreeGo2Env()
    make_policy, params, metadata = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )

    obs_size = env.observation_size
    act_size = env.action_size

    try:
        state_obs_shape = obs_size["state"]
        privileged_obs_shape = obs_size["privileged_state"]
        
        state_input_shape = state_obs_shape
        dummy_privileged_shape = privileged_obs_shape
        
        print(f"State input shape: {state_input_shape}")
        print(f"Privileged state shape: {dummy_privileged_shape}")

    except Exception as e:
        print(f"Error: Could not determine observation shapes from env.observation_size: {e}")
        print("Assuming env.observation_size is flat.")
        state_input_shape = (env.observation_size[0])
        return

    example_input_array = jnp.zeros(state_input_shape, dtype=jnp.float32)

    jax_policy_fn = make_policy(params, deterministic=True)

    def deterministic_policy_wrapper(state_obs):

        dummy_key = jax.random.PRNGKey(0)
        dummy_privileged_state = jnp.zeros(dummy_privileged_shape, dtype=jnp.float32)

        obs_dict = {
            'state': state_obs,
            'privileged_state': dummy_privileged_state
        }

        actions, _ = jax_policy_fn(obs_dict, dummy_key)
        return actions

    jitted_wrapper = jax.jit(deterministic_policy_wrapper)

    print(f"Testing JAX wrapper with input shape: {example_input_array.shape}")
    jax_output = jitted_wrapper(example_input_array)
    print(f"JAX wrapper output shape: {jax_output.shape}. (Expected: (1, {act_size}))")

    output_dir = "onnx_models"
    output_path = f"{output_dir}/{FLAGS.checkpoint_name}_jax.onnx"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConverting JAX model to ONNX at {output_path}...")
    
    jax2onnx.to_onnx(
        fn=jitted_wrapper,
        inputs=[example_input_array],
        opset=11,
        return_mode="file",
        output_path=output_path,
        model_name=f"{FLAGS.checkpoint_name}_jax",
    )
    
    print("-----------------------------------------")
    print(f"Successfully converted JAX model to ONNX!")
    print(f"Model saved to: {output_path}")
    print("-----------------------------------------")


if __name__ == '__main__':
    app.run(main)
