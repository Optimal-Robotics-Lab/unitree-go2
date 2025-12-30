from absl import app

import jax
import jax.numpy as jnp

from training.algorithms.ppo import agent
from training import statistics


jax.config.update("jax_enable_x64", True)


def main(argv=None):
    del argv  # Unused.

    observation = {"state": jnp.ones((51,)), "priviledge_state": jnp.ones((101,))}
    observation_size = {
        "state": observation["state"].shape[0],
        "priviledge_state": observation["priviledge_state"].shape[0],
    }
    action_size = 12

    policy_input_normalization = statistics.RunningStatistics(
        reference_input=observation["state"],
    )
    value_input_normalization = statistics.RunningStatistics(
        reference_input=observation["priviledge_state"],
    )

    model = agent.Agent(
        observation_size=observation_size,
        action_size=action_size,
        policy_input_normalization=policy_input_normalization,
        value_input_normalization=value_input_normalization,
        policy_observation_key="state",
        value_observation_key="priviledge_state",
    )

    print(model)

    print("Agent model created successfully.")


if __name__ == "__main__":
    app.run(main)
