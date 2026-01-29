import functools
from absl import app, flags

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx
import optax

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
    mjx_model = mjx.put_model(mj_model)
    control_rate = 0.02

    num_time_steps = 500
    max_switches = 10
    trajectory_time = num_time_steps * control_rate

    # Probability of Step Function Trajectory
    step_function_prob = 0.2
    minimum_duration_between_steps = 1.0
    minimum_step_duration = int(
        minimum_duration_between_steps / control_rate
    )

    # Model Site IDs and Home Position:
    home_position = jnp.array(mj_model.keyframe('home').qpos)
    base_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
    )
    feet_site = [
        'front_right_foot',
        'front_left_foot',
        'hind_right_foot',
        'hind_left_foot',
    ]
    feet_site_idx = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    assert not any(id_ == -1 for id_ in feet_site_idx), 'Site not found.'
    foot_site_ids = np.array(feet_site_idx)

    # Initialize IK Optimizer:
    learning_rate = 0.05
    optimizer = optax.adam(learning_rate)

    def generate_ik_trajectory(
        key: jax.Array,
        num_time_steps: int = 500,
    ) -> jnp.ndarray:
        """
        Generates a joint-space trajectory (T, n_joints) by tracking
        safe random Cartesian targets using Gradient Descent IK.
        """

        # Task Space Safe Zones for Feet:
        half_size = jnp.array([0.2, 0.1, 0.15])
        center_wrt_base = jnp.array([
            [0.2, -0.15, -0.2,],    # Front Right
            [0.2,  0.15, -0.2,],    # Front Left
            [-0.2, -0.15, -0.2,],   # Rear Right
            [-0.2,  0.15, -0.2,],   # Rear Left
        ])
        lb = center_wrt_base - half_size[None, :]   # (4, 3) Lower Bounds
        ub = center_wrt_base + half_size[None, :]   # (4, 3) Upper Bounds
        bound_range = ub - lb

        def generate_step_targets(
            rng: jax.Array, max_switches: int = 10, minimum_step_duration: int = 20,
        ) -> jnp.ndarray:
            key_switches, key_deltas, key_values = jax.random.split(rng, 3)

            # Random number of switches:
            n_switches = jax.random.randint(
                key_switches, shape=(), minval=1, maxval=max_switches + 1,
            )

            # Generate random gaps
            deltas = jax.random.randint(
                key_deltas,
                shape=(max_switches,),
                minval=0,
                maxval=100,
            )

            # Enforce Minimum Duration:
            safe_deltas = deltas + minimum_step_duration

            # Convert Deltas to Timestamps:
            generated_times = jnp.cumsum(safe_deltas)

            # Masking Logic:
            idx = jnp.arange(max_switches)
            count_mask = idx < n_switches

            # Filter by Maximum Time:
            time_mask = generated_times < num_time_steps

            # Combine masks: Valid if index < N AND time < MaxTime
            valid_mask = count_mask & time_mask

            # Active Switch Times:
            active_switch_times = jnp.where(
                valid_mask, generated_times, num_time_steps + 1,
            )

            # Generate Values:
            param_shape = (max_switches + 1, *lb.shape)
            values = jax.random.uniform(
                key_values, shape=param_shape, minval=lb, maxval=ub
            )

            # Task Space Trajectory:
            t = jnp.arange(num_time_steps)
            indices = jnp.searchsorted(active_switch_times, t, side='right')
            trajectory = values[indices]

            return trajectory

        def generate_sinusoidal_targets(key: jax.Array) -> jnp.ndarray:
            key, frequency_key, phase_key, amplitude_key, center_key = jax.random.split(key, 5)
            period_lb, period_ub = 0.5, 10.0
            frequency_lb, frequency_ub = 2 * jnp.pi / period_ub, 2 * jnp.pi / period_lb
            frequency = jax.random.uniform(
                frequency_key, shape=(4, 3), minval=frequency_lb, maxval=frequency_ub,
            )
            phase = jax.random.uniform(
                phase_key, shape=(4, 3), minval=0.0, maxval=2*jnp.pi,
            )
            amplitude = jax.random.uniform(
                amplitude_key, shape=(4, 3), minval=-bound_range, maxval=bound_range,
            )
            centers = jax.random.uniform(
                center_key, shape=(4, 3), minval=lb, maxval=ub,
            )

            # Create Task Space Trajectory
            t = jnp.linspace(0, trajectory_time, num_time_steps)[:, None, None]
            targets = centers[None, :, :] + amplitude[None, :, :] * jnp.sin(
                frequency[None, :, :] * t + phase[None, :, :]
            )

            return targets

        # Choose Target Generation Method
        key, target_key, bernoulli_key = jax.random.split(key, 3)
        step_targets = generate_step_targets(target_key, max_switches=max_switches, minimum_step_duration=minimum_step_duration)
        sinusoidal_targets = generate_sinusoidal_targets(target_key)
        mask = jax.random.bernoulli(bernoulli_key, p=step_function_prob)
        targets = jnp.where(
            mask,
            step_targets,
            sinusoidal_targets,
        )

        # Clip to Safe Zones:
        targets = jnp.clip(targets, lb[None, ...], ub[None, ...])

        # Inverse Kinematics:
        def fk_fn(q):
            # Update Physics
            d = mjx.make_data(mjx_model)
            d = d.replace(qpos=q)
            d = mjx.kinematics(mjx_model, d)

            # Positions in World Frame
            feet_pos = d.site_xpos[foot_site_ids]
            base_pos = d.xpos[base_id]

            # Position Relative to Base
            return feet_pos - base_pos

        # Loss Function: Distance to Target + Regularization
        def ik_loss(q, target_pos):
            current_pos = fk_fn(q)
            dist_error = jnp.sum((current_pos - target_pos) ** 2)
            reg_error = jnp.sum((q - home_position) ** 2)
            return dist_error + 0.05 * reg_error

        # Gradient Step
        grad_fn = jax.value_and_grad(ik_loss)

        def solve_step(carry_q, target):
            """
                Solves IK for a single timestep using an inner optimization loop.
                carry_q: The solution from the previous timestep (Warm start)
                target: The Cartesian target for the current timestep
            """

            # Initialize Optimizer State for this timestep
            opt_state = optimizer.init(carry_q)

            # Inner Optimization Loop
            def optimization_loop(carry, unused_t):
                q, opt_state = carry

                # Calculate Gradients
                loss, grads = grad_fn(q, target)
                updates, opt_state = optimizer.update(grads, opt_state, params=q)
                q_new = optax.apply_updates(q, updates)

                # Project back to Joint Limits (Safety)
                q_new = jnp.clip(
                    q_new, mj_model.jnt_range[:, 0], mj_model.jnt_range[:, 1],
                )
                return (q_new, opt_state), None

            (q_solved, _), _ = jax.lax.scan(
                optimization_loop,
                init=(carry_q, opt_state),
                xs=None,
                length=50,
            )

            # Return the result for the trajectory scan
            return q_solved, q_solved

        # Warm start and run scan:
        init_q = home_position
        final_q, q_trajectory = jax.lax.scan(solve_step, init_q, targets)

        return q_trajectory

    # Generate Trajectories:
    def loop(carry, unused_t):
        key = carry

        # Generate Trajectory Based on Mask
        key, trajectory_key = jax.random.split(key)
        trajectory = functools.partial(
            generate_ik_trajectory, num_time_steps=num_time_steps,
        )(trajectory_key)

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

    # Flatten and Save: (Trials, Time, Joints) -> (Trials * Time, Joints)
    num_joints = trajectories.shape[-1]
    flattened_data = np.reshape(np.array(trajectories), (-1, num_joints))
    np.savetxt(output_path, flattened_data)


if __name__ == '__main__':
    app.run(main)
