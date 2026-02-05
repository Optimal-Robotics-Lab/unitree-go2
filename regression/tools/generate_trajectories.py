from absl import app, flags

from typing import Tuple
import functools

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
import mujoco.viewer
from mujoco import mjx

import optax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')


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
    step_function_prob = 1.0
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
            Generates a joint-space trajectory (T, n_joints) by tracking
            safe random Cartesian targets using Gradient Descent IK.
        """

        # Safe Task Space Bounds:
        half_size = jnp.array([0.2, 0.2, 0.3])
        center_wrt_base = jnp.array([
            [0.2, -0.25, -0.15,],    # Front Right
            [0.2,  0.25, -0.15,],    # Front Left
            [-0.2, -0.25, -0.15,],   # Rear Right
            [-0.2,  0.25, -0.15,],   # Rear Left
        ])
        lb = center_wrt_base - half_size[None, :]   # (4, 3) Lower Bounds
        ub = center_wrt_base + half_size[None, :]   # (4, 3) Upper Bounds

        # Generate Joint-Space Chirp Targets:
        def generate_joint_chirps(key):
            # Amplitude Profile: [Hip, Thigh, Calf]
            profile = jnp.array([0.2, 0.3, 0.6] * 4)

            # Randomize Frequencies per joint:
            key, frequency_key = jax.random.split(key)
            f_start = 0.1
            f_end = jax.random.uniform(frequency_key, shape=(12,), minval=2.0, maxval=4.0)

            # Chirp Signal
            t = jnp.linspace(0, trajectory_time, num_time_steps)[:, None]
            k = (f_end - f_start) / trajectory_time
            phase = 2 * jnp.pi * (f_start * t + (k / 2) * t**2)

            # Randomize Phase Offsets:
            key, phase_key = jax.random.split(key)
            offsets = jax.random.uniform(phase_key, shape=(12,), minval=0, maxval=2*jnp.pi)

            # Randomize Amplitude Scaling:
            key, amp_key = jax.random.split(key)
            amplitude_scale = jax.random.uniform(amp_key, shape=(12,), minval=0.01, maxval=1.0)
            amplitude_scale = jnp.sqrt(amplitude_scale)
            amplitude = profile * amplitude_scale

            # Calculate Desired Joint Trajectory
            q_desired = home_position[None, :] + \
                amplitude[None, :] * jnp.sin(phase + offsets[None, :])

            return q_desired

        def generate_step_targets(
            rng: jax.Array, max_switches: int = 10, minimum_step_duration: int = 20,
        ) -> jnp.ndarray:
            # Task Space Safe Zones for Step Targets:
            half_size = jnp.array([0.2, 0.1, 0.15])
            center_wrt_base = jnp.array([
                [0.2, -0.15, -0.2,],    # Front Right
                [0.2,  0.15, -0.2,],    # Front Left
                [-0.2, -0.15, -0.2,],   # Rear Right
                [-0.2,  0.15, -0.2,],   # Rear Left
            ])
            lb = center_wrt_base - half_size[None, :]
            ub = center_wrt_base + half_size[None, :]

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

        # Forward Kinematics Function:
        def fk_fn(q):
            d = mjx.make_data(mjx_model)
            d = d.replace(qpos=q)
            d = mjx.kinematics(mjx_model, d)

            # Positions in World Frame
            feet_pos = d.site_xpos[foot_site_ids]
            base_pos = d.xpos[base_id]

            # Position Relative to Base
            return feet_pos - base_pos

        # Loss Function: Distance to Target + Regularization
        def ik_loss(q, target_pos, target_qpos, weight):
            current_pos = fk_fn(q)
            dist_error = jnp.sum((current_pos - target_pos) ** 2)
            reg_error = jnp.sum((q - target_qpos) ** 2)
            return dist_error + weight * reg_error

        def solve_step(carry, target):
            """
                Solves IK for a single timestep using an inner optimization loop.
                carry: The solution from the previous timestep (Warm start)
                target: The Cartesian and Joint targets for the current timestep
            """
            carry_q, _ = carry
            target_pos, target_q = target

            # Initialize Optimizer State for this timestep
            opt_state = optimizer.init(carry_q)

            # Inner Optimization Loop
            def optimization_loop(carry, unused_t):
                q, opt_state = carry

                # Calculate Gradients
                loss, grads = grad_fn(q, target_pos, target_q)
                updates, opt_state = optimizer.update(grads, opt_state, params=q)
                q_new = optax.apply_updates(q, updates)

                # Project back to Joint Limits:
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
            return (q_solved, q_solved), q_solved

        # Choose Target Generation Method
        key, target_key, bernoulli_key = jax.random.split(key, 3)
        chirp_joint_targets = generate_joint_chirps(target_key)
        chirp_taskspace_targets = jax.vmap(fk_fn)(chirp_joint_targets)
        chirp_taskspace_targets = jnp.clip(
            chirp_taskspace_targets,
            lb[None, ...],
            ub[None, ...],
        )
        step_taskspace_targets = generate_step_targets(
            target_key, max_switches=max_switches, minimum_step_duration=minimum_step_duration,
        )
        mask = jax.random.bernoulli(bernoulli_key, p=step_function_prob)
        targets = jnp.where(
            mask,
            step_taskspace_targets,
            chirp_taskspace_targets,
        )

        # Corresponding Regularization Joint Targets
        joint_targets = jnp.where(
            mask,
            home_position,
            chirp_joint_targets,
        )

        # Corresponding Regularization Weights
        weight = jnp.where(
            mask,
            0.05,
            0.1,
        )

        # Gradient Step
        loss_fn = functools.partial(ik_loss, weight=weight)
        grad_fn = jax.value_and_grad(loss_fn)

        # Warm start and run scan:
        init_q = (home_position, home_position)
        final_q, q_trajectory = jax.lax.scan(
            solve_step,
            init_q,
            (targets, joint_targets),
        )

        return q_trajectory, targets

    # Generate Trajectories:
    def loop(carry, unused_t):
        key = carry

        # Generate Trajectory Based on Mask
        key, trajectory_key = jax.random.split(key)
        qpos_trajectory, target_trajectory = functools.partial(
            generate_ik_trajectory, num_time_steps=num_time_steps,
        )(trajectory_key)

        return key, (qpos_trajectory, target_trajectory)

    key = jax.random.key(FLAGS.seed)
    _, (trajectories, target_trajectories) = jax.lax.scan(
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
                for qpos_trajectory, target_trajectory in zip(trajectories, target_trajectories):
                    for t in range(num_time_steps):
                        data.qpos = np.array(qpos_trajectory[t])
                        mujoco.mj_forward(mj_model, data)

                        # Draw Targets:
                        base_pos = data.xpos[base_id]
                        base_mat = data.xmat[base_id].reshape(3, 3)

                        targets_local = target_trajectory[t]
                        targets_world = base_pos + (targets_local @ base_mat.T)

                        # Draw Spheres
                        viewer.user_scn.ngeom = 0
                        for foot_idx in range(4):
                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[foot_idx],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.02, 0, 0],
                                pos=targets_world[foot_idx],
                                mat=np.eye(3).flatten(),
                                rgba=[1, 0, 0, 0.8]
                            )

                        viewer.user_scn.ngeom = 4

                        viewer.sync()
                        time.sleep(control_rate)

                        if not viewer.is_running():
                            break
                termination_flag = True

    # Save Trajectories:
    data_directory = os.path.join(
        os.path.dirname(__file__), 'data/generated_trajectories',
    )
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    output_path = os.path.join(
        data_directory,
        f'{FLAGS.output_filename}.csv',
    )

    # Flatten and Save: (Trials, Time, Joints) -> (Trials * Time, Joints)
    num_joints = trajectories.shape[-1]
    header = f"SHAPE:{FLAGS.num_trajectories},{num_time_steps},{num_joints}"
    flattened_data = np.reshape(np.array(trajectories), (-1, num_joints))
    np.savetxt(output_path, flattened_data, delimiter=',', header=header, comments='')


if __name__ == '__main__':
    app.run(main)
