from typing import Callable, List, Tuple, Optional
import warnings
import os
from pathlib import Path
from dataclasses import dataclass
import time

import jax
import numpy as np

import mujoco

import brax
from brax.io import html, mjcf
from brax.envs.wrappers.training import EvalWrapper

import training.module_types as types
from training.training_utilities import unroll_policy_trajectory


@dataclass
class RenderOptions:
    filepath: str
    render_interval: int = 5
    duration: float = 10.0


class Evaluator:

    def __init__(
        self,
        env: types.Env,
        num_envs: int,
        episode_length: int,
        action_repeat: int,
        key: types.PRNGKey,
        render_options: Optional[RenderOptions] = None,
    ):
        self.key = key
        self.walltime = 0.0
        self.mj_model = env.mj_model
        self.dt = env.step_dt

        # Create Brax System for HTML rendering:
        sys = mjcf.load_model(self.mj_model)
        self.sys = sys.tree_replace({'opt.timestep': env.dt})
        self.dt = env.dt

        env = EvalWrapper(env)

        self.render = False if render_options is None else True
        if render_options is not None:
            self.filepath = os.path.join(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
                f"visualization/{render_options.filepath}",
            )
            Path(self.filepath).mkdir(parents=True, exist_ok=True)
            self.current_filepath = self.filepath

            if render_options.duration <= 0:
                raise ValueError(
                    f"Duration must be positive, got {render_options.duration}."
                )

            self.render_interval = render_options.render_interval
            self.duration = render_options.duration

            total_sim_steps = episode_length // action_repeat
            render_sim_steps = int(np.ceil(self.duration / self.dt))
            if render_sim_steps > total_sim_steps:
                warnings.warn(
                    f"Requested render duration ({self.duration}s) exceeds the total simulation steps ({total_sim_steps * self.dt}s). "
                    f"Adjusting render duration to {total_sim_steps * self.dt}s."
                )
            self.render_episode_length = render_sim_steps if render_sim_steps < total_sim_steps else total_sim_steps

        def _evaluation_loop(
            policy_fn: Callable[[types.Observation, types.PRNGKey], types.Action],
            key: types.PRNGKey,
        ) -> types.State:
            reset_keys = jax.random.split(key, num_envs)
            initial_state = env.reset(reset_keys)
            final_state, states = unroll_policy_trajectory(
                env,
                initial_state,
                policy_fn,
                key,
                num_steps=episode_length // action_repeat,
            )
            return final_state, states

        self.evaluation_loop = _evaluation_loop
        self.steps_per_epoch = episode_length * num_envs

    def evaluate(
        self,
        policy_fn: Callable[[types.Observation, types.PRNGKey], types.Action],
        training_metrics: types.Metrics,
        iteration: int,
        aggregate_episodes: bool = True,
    ) -> types.Metrics:
        self.key, subkey = jax.random.split(self.key)

        start_time = time.time()
        state, states = jax.jit(
            self.evaluation_loop,
        )(policy_fn, subkey)
        evaluation_metrics = state.info['eval_metrics']
        evaluation_metrics.active_episodes.block_until_ready()
        epoch_time = time.time() - start_time

        # Render the states if the render method is defined:
        self.render_flag = (iteration % self.render_interval == 0)
        if self.render and self.render_flag:
            self._render_html(states, iteration)

        metrics = {}
        for func in [np.mean, np.std]:
            suffix = '_std' if func == np.std else ''
            metrics.update({
                f'eval/episode_{name}{suffix}': (
                    func(value) if aggregate_episodes else value
                )
                for name, value in evaluation_metrics.episode_metrics.items()
            })

        metrics['eval/avg_episode_length'] = np.mean(
            evaluation_metrics.episode_steps,
        )
        metrics['eval/epoch_time'] = epoch_time
        metrics['eval/steps_per_second'] = self.steps_per_epoch / epoch_time
        self.walltime = self.walltime + epoch_time
        metrics = {
            'eval/walltime': self.walltime,
            **training_metrics,
            **metrics,
        }

        return metrics

    def _render_html(
        self,
        states: List[Tuple[jax.Array, jax.Array, jax.Array]],
        iteration: int,
    ) -> None:
        """ Render using Brax HTML renderer. """
        qpos, xpos, xquat = jax.tree.map(lambda x: x[:, 0, :], states)
        data = mujoco.mjx.make_data(self.mj_model)
        data_args = data.__dict__
        data_args['contact'] = brax.mjx.pipeline._reformat_contact(
            self.sys, data.contact,
        )
        state_list = []
        for i in range(self.render_episode_length):
            state_list.append(
                brax.mjx.base.State(
                    q=qpos[i],
                    qd=np.zeros(self.mj_model.nv),
                    x=brax.base.Transform(
                        pos=xpos[i][1:],
                        rot=xquat[i][1:],
                    ),
                    xd=brax.base.Motion(
                        vel=np.zeros_like(data.cvel[1:, 3:]),
                        ang=np.zeros_like(data.cvel[1:, :3]),
                    ),
                    **data_args,
                ),
            )

        html_string = html.render(
            sys=self.sys,
            states=state_list,
            height="100vh",
            colab=False,
        )

        html_string = html.render(
            sys=self.sys,
            states=state_list,
            height="100vh",
            colab=False,
        )

        filepath = os.path.join(self.filepath, f'{iteration}.html')
        self.current_filepath = filepath
        with open(filepath, "w") as f:
            f.writelines(html_string)
