from typing import Callable, List, Tuple, Optional
import warnings
import os
from pathlib import Path
from dataclasses import dataclass, field
import time

import jax
import numpy as np

import mujoco

import brax
from brax.io import html
from brax import envs

import training.module_types as types
from training.training_utilities import unroll_policy_trajectory

import imageio.v3 as iio


"""
    TODO:This needs a refactor
"""


def create_default_camera():
    """Creates a MjvCamera instance with custom default values."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = np.array([0.0, 0.0, 0.5])
    camera.distance = 10.0
    camera.azimuth = 120.0
    camera.elevation = -45.0
    return camera


@dataclass
class RenderOptions:
    filepath: str
    num_envs: int = 4
    render_interval: int = 5
    spacing: float = 1.0
    duration: float = 10.0
    # Video options:
    fps: int = 30
    video_format: str = 'html'
    # MuJoCo Render Options:
    vopt: mujoco.MjvOption = field(default_factory=mujoco.MjvOption)
    pert: mujoco.MjvPerturb = field(default_factory=mujoco.MjvPerturb)
    camera: mujoco.MjvCamera = field(default_factory=create_default_camera)


class Evaluator:

    def __init__(
        self,
        env: envs.Env,
        num_envs: int,
        episode_length: int,
        action_repeat: int,
        key: types.PRNGKey,
        render_options: Optional[RenderOptions] = None,
    ):
        self.key = key
        self.walltime = 0.0
        self.sys = env.sys.tree_replace({'opt.timestep': env.dt})
        self.dt = env.dt

        env = envs.training.EvalWrapper(env)

        self.render_type = None
        if render_options is not None:
            self.filepath = os.path.join(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
                f"visualization/{render_options.filepath}",
            )
            Path(self.filepath).mkdir(parents=True, exist_ok=True)
            self.current_filepath = self.filepath

            if render_options.num_envs > num_envs:
                raise ValueError(
                    f"Requested {render_options.num_envs} environments for rendering, "
                    f"but only {num_envs} are available."
                )
            if render_options.video_format not in ['gif', 'webm', 'mp4', 'html']:
                raise ValueError(
                    f"Unsupported video format: {render_options.video_format}. "
                    "Supported formats are: gif, webm, mp4, html."
                )
            if render_options.duration <= 0:
                raise ValueError(
                    f"Duration must be positive, got {render_options.duration}."
                )

            # Enable rendering:
            self.render_type = 'video' if render_options.video_format in ['gif', 'webm', 'mp4'] else 'html'

            self.num_envs = render_options.num_envs
            self.render_interval = render_options.render_interval
            self.spacing = render_options.spacing
            self.duration = render_options.duration

            # Video format:
            self.fps = render_options.fps
            self.video_format = render_options.video_format

            # Calculate the number of rows and columns for rendering:
            self.num_rows = int(np.sqrt(self.num_envs))
            self.num_cols = int(np.ceil(self.num_envs / self.num_rows))

            # MuJoCo render structs:
            self.vopt = render_options.vopt
            self.pert = render_options.pert
            self.camera = render_options.camera

            # Override the camera settings:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.distance = 5.0 * self.spacing * self.num_cols

            # Preallocate:
            self.mj_datas = [mujoco.MjData(self.sys.mj_model) for _ in range(self.num_envs)]
            x_pos = np.linspace(
                -self.spacing * (self.num_cols - 1) / 2,
                self.spacing * (self.num_cols - 1) / 2,
                self.num_cols,
            )
            y_pos = np.linspace(
                -self.spacing * (self.num_rows - 1) / 2,
                self.spacing * (self.num_rows - 1) / 2,
                self.num_rows
            )
            x, y = np.meshgrid(x_pos, y_pos)
            self.grid_x = x.flatten()[:self.num_envs]
            self.grid_y = y.flatten()[:self.num_envs]

            # Pre-calculate the indices to render from the full trajectory.
            total_sim_steps = episode_length // action_repeat
            render_sim_steps = int(np.ceil(self.duration / self.dt))
            if render_sim_steps > total_sim_steps:
                warnings.warn(
                    f"Requested render duration ({self.duration}s) exceeds the total simulation steps ({total_sim_steps * self.dt}s). "
                    f"Adjusting render duration to {total_sim_steps * self.dt}s."
                )
            self.render_episode_length = render_sim_steps if render_sim_steps < total_sim_steps else total_sim_steps
            sim_steps_per_video_frame = (1.0 / self.fps) / self.dt
            num_video_frames = int(np.ceil(self.render_episode_length * self.dt * self.fps))
            self.indices_to_render = [
                min(int(round(i * sim_steps_per_video_frame)), self.render_episode_length - 1)
                for i in range(num_video_frames)
            ]

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
            self.evaluation_loop, static_argnums=(0,)
        )(policy_fn, subkey)
        evaluation_metrics = state.info['eval_metrics']
        evaluation_metrics.active_episodes.block_until_ready()
        epoch_time = time.time() - start_time

        # Render the states if the render method is defined:
        self.render_flag = (iteration % self.render_interval == 0)
        if self.render_type == 'video' and self.render_flag:
            self._render_video(states, iteration)
        elif self.render_type == 'html' and self.render_flag:
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
        data = mujoco.mjx.make_data(self.sys.mj_model)
        data_args = data.__dict__
        data_args['contact'] = brax.mjx.pipeline._reformat_contact(
            self.sys, data.contact,
        )
        state_list = []
        for i in range(self.render_episode_length):
            state_list.append(
                brax.mjx.base.State(
                    q=qpos[i],
                    qd=np.zeros(self.sys.nv),
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

        filepath = os.path.join(self.filepath, f'{iteration}.{self.video_format}')
        self.current_filepath = filepath
        with open(filepath, "w") as f:
            f.writelines(html_string)

    def _render_video(
        self,
        position_history: List[jax.Array],
        iteration: int,
    ) -> None:
        """Render the position_history of the environments in a grid."""
        position_history = np.asarray(position_history[self.indices_to_render, ...])
        position_history.flags.writeable = True

        frames = []
        with mujoco.Renderer(self.sys.mj_model, height=480, width=640) as renderer:
            for i, position in enumerate(position_history):
                time_start = time.time()
                for j, mj_data in enumerate(self.mj_datas):
                    qpos = position[j]
                    qpos[:2] += np.array([self.grid_x[j], self.grid_y[j]])
                    mj_data.qpos = qpos

                    mujoco.mj_fwdPosition(self.sys.mj_model, mj_data)

                    if j == 0:
                        mujoco.mjv_updateScene(
                            self.sys.mj_model,
                            mj_data,
                            self.vopt,
                            self.pert,
                            self.camera,
                            mujoco.mjtCatBit.mjCAT_ALL,
                            renderer.scene,
                        )
                    else:
                        mujoco.mjv_addGeoms(
                            self.sys.mj_model,
                            mj_data,
                            self.vopt,
                            self.pert,
                            mujoco.mjtCatBit.mjCAT_DYNAMIC,
                            renderer.scene,
                        )

                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

                time_render_start = time.time()
                frame = renderer.render()
                print(f'Render time: {time.time() - time_render_start} seconds')

                frames.append(frame)
                print(f"Iteration {i}: Elapsed time: {time.time() - time_start} seconds")

            print(f"Rendering frame")
            start_time = time.time()
            filepath = os.path.join(self.filepath, f'{iteration}.{self.video_format}')
            self.current_filepath = filepath
            kwargs = {'fps': self.fps}
            if self.video_format == 'mp4':
                kwargs['codec'] = 'libx264'
                kwargs['out_pixel_format'] = 'yuv420p'
            elif self.video_format == 'webm':
                kwargs['codec'] = 'libvpx-vp9'
            iio.imwrite(filepath, frames, plugin='pyav', is_batch=True, **kwargs)
            print(f"Video Render Elapsed time: {time.time() - start_time} seconds")
