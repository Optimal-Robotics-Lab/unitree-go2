import os
from absl import app, flags

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import flax
from flax import nnx

from training.envs.unitree_go2 import unitree_go2_joystick
from training.envs.unitree_go2 import config

import training.statistics as statistics
from training.algorithms.ppo import agent
from training import checkpoint_utilities

import jax2onnx

import onnx
from onnxsim import simplify


os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)


def main(argv=None):
    # Setup Environments:
    scene = 'scene_mjx.xml'
    environment_config = config.EnvironmentConfig(
        filename=scene,
        action_scale=0.5,
        control_timestep=0.02,
        optimizer_timestep=0.004,
    )
    env = unitree_go2_joystick.UnitreeGo2Env(
        environment_config=environment_config,
    )

    observation_size = env.observation_size
    action_size = env.action_size
    reference_observation = {
        key: jnp.zeros(value) for key, value in observation_size.items()
    }

    # Setup agent:
    policy_layer_size = [512, 256, 128,]
    value_layer_size = [512, 256, 128,]
    activation_fn = jax.nn.swish
    policy_kernel_init = jax.nn.initializers.lecun_uniform()
    value_kernel_init = jax.nn.initializers.variance_scaling(
        scale=0.01, mode="fan_in", distribution="uniform",
    )
    policy_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["state"],
    )
    value_input_normalization = statistics.RunningStatistics(
        reference_input=reference_observation["privileged_state"],
    )
    model = agent.Agent(
        observation_size=observation_size,
        action_size=action_size,
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

    # Restore checkpoint:
    restore_directory = os.path.join(
        os.path.dirname(__file__),
        f"checkpoints/{FLAGS.checkpoint_name}",
    )
    restore_manager = checkpoint_utilities.create_checkpoint_manager(
        checkpoint_directory=restore_directory,
    )
    restored_checkpoint, _ = checkpoint_utilities.restore_training_state(
        manager=restore_manager,
        agent=model,
        iteration=FLAGS.checkpoint_iteration,
    )

    nnx.update(model, restored_checkpoint.agent)

    def inference_wrapper(observation: jax.Array) -> jax.Array:
        dummy_key = jax.random.key(0)

        actions, info = model.get_actions(
            observation, 
            dummy_key, 
            deterministic=True
        )

        return actions

    reference_input = reference_observation["state"]

    model_proto = jax2onnx.to_onnx(
        fn=inference_wrapper,
        inputs=[reference_input],
        opset=17,
        return_mode="proto",
        model_name=f"{FLAGS.checkpoint_name}_jax",
        enable_double_precision=False,
    )

    # jax2onnx does not currently support naming inputs and outputs directly:
    def rename_io(model_proto, input_names: list[str], output_names: list[str]):
        """
            Name the input and output nodes of the ONNX model.
        """
        graph = model_proto.graph
        
        for i, new_name in enumerate(input_names):
            if i >= len(graph.input): break
            old_name = graph.input[i].name
            graph.input[i].name = new_name
            for node in graph.node:
                for input_idx, input_name in enumerate(node.input):
                    if input_name == old_name:
                        node.input[input_idx] = new_name

        for i, new_name in enumerate(output_names):
            if i >= len(graph.output): break
            old_name = graph.output[i].name
            graph.output[i].name = new_name
            for node in graph.node:
                for output_idx, output_name in enumerate(node.output):
                    if output_name == old_name:
                        node.output[output_idx] = new_name

        return model_proto

    model_proto = rename_io(
        model_proto, 
        input_names=["observations"], 
        output_names=["actions"]
    )

    # Simplify ONNX model: jax2onnx produces models with redundant/dead nodes
    model_simp, check = simplify(model_proto)

    assert check, "Simplified ONNX model could not be validated"

    output_dir = "onnx_models"
    output_path = f"{output_dir}/{FLAGS.checkpoint_name}_jax.onnx"
    os.makedirs(output_dir, exist_ok=True)

    onnx.save_model(model_simp, output_path)


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_name')
    app.run(main)
