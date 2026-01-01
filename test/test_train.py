import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
from absl import app, logging

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import distrax
import optax

import orbax.checkpoint as ocp

import wandb

from training import statistics
from training.algorithms.ppo import train
from training.algorithms.ppo import agent

from training.envs.unitree_go2 import unitree_go2_joystick

from training import checkpoint_utilities
from training import metrics_utilities

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

logging.set_verbosity(logging.FATAL)


def main(argv=None):
    # Setup environment:
    env = unitree_go2_joystick.UnitreeGo2Env()
    eval_env = unitree_go2_joystick.UnitreeGo2Env()

    observation_size = env.observation_size
    action_size = env.action_size
    reference_observation = {key: jnp.zeros(value) for key, value in observation_size.items()}

    print(f"Observation size: {observation_size}")
    print(f"Action size: {action_size}")

    # Setup agent:
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
        policy_observation_key="state",
        value_observation_key="privileged_state",
    )

    print(f"Agent Model: \n {model}")

    render_options = metrics_utilities.RenderOptions(
        filepath="test",
        render_interval=5,
        duration=10.0,
    )

    train_fn = functools.partial(
        train.train,
        environment=env,
        evaluation_environment=eval_env,
        num_epochs=1,
        num_training_steps=1,
        episode_length=1,
        num_envs=2,
        num_evaluation_envs=2,
        render_options=render_options,
    )
    policy, metrics = train_fn(agent=model)

    print("Training completed.")
    print("Final Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    app.run(main)
