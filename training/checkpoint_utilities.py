import os
from typing import Any, Optional, Dict
import dataclasses

import flax.struct
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp


@dataclasses.dataclass
class RestoredCheckpoint:
    agent: nnx.Module
    opt_state: optax.OptState | None
    env_steps: int
    metadata: Dict[str, Any]


def create_checkpoint_manager(
    checkpoint_directory: str,
    max_to_keep: int = 5,
    save_interval_steps: int = 1,
) -> ocp.CheckpointManager:
    """
        Creates a Orbax CheckpointManager with StandardCheckpointer.
    """
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
        create=True,
    )

    return ocp.CheckpointManager(
        os.path.abspath(checkpoint_directory),
        options=options,
    )


def save_checkpoint(
    manager: ocp.CheckpointManager,
    iteration: int,
    agent: nnx.Module,
    opt_state: optax.OptState,
    env_steps: int,
    **metadata: Any,
) -> None:
    """
        Saves the agent, optimizer, and metadata using StandardSave.
    """
    # Extract State from NNX Agent (Parameters + Stats)
    _, agent_state = nnx.split(agent)

    payload = {
        'agent': agent_state,
        'opt_state': opt_state,
        'env_steps': env_steps,
        'metadata': metadata,
    }

    manager.save(
        iteration,
        args=ocp.args.StandardSave(item=payload),
    )


def restore_checkpoint(
    manager: ocp.CheckpointManager,
    agent: nnx.Module,
    opt_state: Optional[optax.OptState] = None,
    iteration: Optional[int] = None,
) -> Optional[RestoredCheckpoint]:
    """
        Restores the checkpoint directly into the provided Agent instance.
    """
    if iteration is None:
        iteration = manager.latest_step()

    if iteration is None:
        return None

    # Get Target Payload Structure:
    _, agent_state_target = nnx.split(agent)

    target_payload = {
        'agent': agent_state_target,
        'env_steps': 0,
        'metadata': {},
    }

    if opt_state is not None:
        target_payload['opt_state'] = opt_state

    restored = manager.restore(
        iteration,
        args=ocp.args.StandardRestore(item=target_payload)
    )

    # Restore Agent:
    nnx.update(agent, restored['agent'])

    return RestoredCheckpoint(
        agent=agent,
        opt_state=restored.get('opt_state'),
        env_steps=restored.get('env_steps', 0),
        metadata=restored.get('metadata', {})
    )


@flax.struct.dataclass
class agent_metadata:
    agent: str


@flax.struct.dataclass
class loss_metadata:
    policy_clip_coef: float
    value_clip_coef: float
    value_coef: float
    entropy_coef: float
    gamma: float
    gae_lambda: float
    normalize_advantages: bool


@flax.struct.dataclass
class training_metadata:
    num_epochs: int
    num_training_steps: int
    episode_length: int
    num_policy_steps: int
    action_repeat: int
    num_envs: int
    num_evaluation_envs: int
    num_evaluations: int
    deterministic_evaluation: bool
    reset_per_epoch: bool
    seed: int
    batch_size: int
    num_minibatches: int
    num_ppo_iterations: int
    normalize_observations: bool
    optimizer: optax.GradientTransformation | str
    has_adaptive_kl_scheduler: bool


def empty_agent_metadata() -> agent_metadata:
    return agent_metadata(
        agent='',
    )


def empty_loss_metadata() -> loss_metadata:
    return loss_metadata(
        policy_clip_coef=0.0,
        value_clip_coef=0.0,
        value_coef=0.0,
        entropy_coef=0.0,
        gamma=0.0,
        gae_lambda=0.0,
        normalize_advantages=False,
    )


def empty_training_metadata() -> training_metadata:
    return training_metadata(
        num_epochs=0,
        num_training_steps=0,
        episode_length=0,
        num_policy_steps=0,
        action_repeat=0,
        num_envs=0,
        num_evaluation_envs=0,
        num_evaluations=0,
        deterministic_evaluation=False,
        reset_per_epoch=False,
        seed=0,
        batch_size=0,
        num_minibatches=0,
        num_ppo_iterations=0,
        normalize_observations=False,
        optimizer='',
        has_adaptive_kl_scheduler=False,
    )
