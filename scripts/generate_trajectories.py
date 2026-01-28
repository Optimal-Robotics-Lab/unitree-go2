import functools
from absl import app, flags

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import mujoco

jax.config.update('jax_enable_x64', True)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'output_filename',
    'trajectories',
    'CSV file to save the generated trajectories to.',
)
flags.DEFINE_integer(
    'seed',
    42,
    'Random Seed for trajectory generation.',
)
flags.DEFINE_integer(
    'num_trajectories',
    50,
    'Number of trajectories to generate.',
)
flags.DEFINE_boolean(
    'view_trajectories',
    False,
    'Whether to visualize the generated trajectories.',
)


def main(argv=None):
    filename = 'mjcf/scene_mjx.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )

    mj_model = mujoco.MjModel.from_xml_path(
        filepath,
    )
    control_rate = 0.02

    num_time_steps = 500
    max_switches = 10
    trajectory_time = num_time_steps * control_rate
    period_lb, period_ub = 1.0, 10.0
    frequency_lb = 2 * np.pi / period_ub
    frequency_ub = 2 * np.pi / period_lb

    # Get Joint Limits:
    joint_lb, joint_ub = mj_model.jnt_range.T
    num_joints = joint_lb.shape[0]

    # Soft Joint Limits:
    limit_factor = 0.5
    joint_center = (joint_lb + joint_ub) / 2
    joint_range = joint_ub - joint_lb
    soft_range = limit_factor * joint_range
    soft_lb = joint_center - soft_range
    soft_ub = joint_center + soft_range

    # Probability of Step Function Trajectory
    step_function_prob = 0.9

    def generate_sinusoid_trajectory(
        key: jax.Array,
        num_time_steps: int = 500,
    ) -> jax.Array:
        key, offset_key, amplitude_key, frequency_key = jax.random.split(key, 4)

        # Generate Sinusoid Parameters
        offset = jax.random.uniform(
            offset_key, shape=(num_joints,), minval=soft_lb, maxval=soft_ub,
        )
        amplitude = jax.random.uniform(
            amplitude_key, shape=(num_joints,), minval=-soft_range, maxval=soft_range,
        )
        frequency = jax.random.uniform(
            frequency_key, shape=(num_joints,), minval=frequency_lb, maxval=frequency_ub,
        )

        # Generate Trajectory
        x = jnp.linspace(0, trajectory_time, num_time_steps)[:, None]
        p_offset = offset[None, :]
        p_amp = amplitude[None, :]
        p_freq = frequency[None, :]
        trajectory = p_offset + p_amp * jnp.sin(p_freq * x)

        # Clip to Joint Limits:
        trajectory = jnp.clip(trajectory, soft_lb, soft_ub)

        return trajectory

    def generate_step_trajectory(
        key: jax.Array,
        num_time_steps: int = 500,
        max_switches: int = 10,
    ) -> jnp.ndarray:
        k_count, k_time, k_val = jax.random.split(key, 3)

        # Calculate number of switches:
        n_switches = jax.random.randint(
            k_count, shape=(), minval=1, maxval=max_switches + 1
        )

        # Generate the Switch Times
        times = jax.random.randint(
            k_time, shape=(max_switches,), minval=1, maxval=num_time_steps
        )
        sorted_times = jnp.sort(times)

        idx = jnp.arange(max_switches)
        mask = idx < n_switches

        active_switch_times = jnp.where(mask, sorted_times, num_time_steps + 1)

        # Generate the Setpoint Values
        param_shape = (max_switches + 1, soft_lb.shape[0])
        values = jax.random.uniform(
            k_val, shape=param_shape, minval=soft_lb, maxval=soft_ub
        )

        # Build the Trajectory:
        t = jnp.arange(num_time_steps)
        indices = jnp.searchsorted(active_switch_times, t, side='right')
        trajectory = values[indices]

        return trajectory

    # Generate Trajectories:
    def loop(carry, unused_t):
        key = carry
        key, subkey = jax.random.split(key)
        mask = jax.random.bernoulli(subkey, p=step_function_prob)

        # Generate Trajectory Based on Mask
        key, trajectory_key = jax.random.split(key)
        trajectory = jax.lax.cond(
            mask,
            functools.partial(generate_step_trajectory, num_time_steps=num_time_steps, max_switches=max_switches),
            functools.partial(generate_sinusoid_trajectory, num_time_steps=num_time_steps),
            trajectory_key,
        )

        return key, trajectory

    key = jax.random.key(FLAGS.seed)
    _, trajectories = jax.lax.scan(
        loop,
        init=key,
        xs=None,
        length=FLAGS.num_trajectories,
    )

    # Play trajectories to check of collisions:
    collisions = 0
    for i, trajectory in enumerate(trajectories):
        # Initialize Mujoco Data:
        data = mujoco.MjData(mj_model)
        for t in range(num_time_steps):
            data.qpos = np.array(trajectory[t])
            mujoco.mj_forward(mj_model, data)
            if data.ncon > 0:
                collisions += 1
                print(f"Collision detected at trajecotry {i} and time step {t}.")

    print(f"Total Collisions Detected: {collisions} out of {FLAGS.num_trajectories} trajectories.")

    if FLAGS.view_trajectories:
        # Visualize Trajectories:
        termination_flag = False
        data = mujoco.MjData(mj_model)
        with mujoco.viewer.launch_passive(mj_model, data) as viewer:
            viewer.cam.trackbodyid = 1
            viewer.cam.distance = 5
            while viewer.is_running() and not termination_flag:
                for trajectory in trajectories:
                    for t in range(num_time_steps):
                        data.qpos = np.array(trajectory[t])
                        mujoco.mj_forward(mj_model, data)
                        viewer.sync()
                        time.sleep(control_rate)
                        if not viewer.is_running():
                            break
                termination_flag = True

    # Save Trajectories:
    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    output_path = os.path.join(
        data_directory,
        f'{FLAGS.output_filename}.csv',
    )
    np.savetxt(output_path, np.array(trajectories))


if __name__ == '__main__':
    app.run(main)
