import os
from typing import Any, Optional, Dict
import dataclasses

import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp


@dataclasses.dataclass
class RestoredCheckpoint:
    agent: nnx.Module
    opt_state: optax.OptState
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
