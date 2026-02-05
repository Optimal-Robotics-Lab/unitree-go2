import jax
import jax.numpy as jnp

import mujoco

import optax


def generate_ik_trajectory(
    key: jax.Array,
    mj_model: mujoco.MjModel,
    trajectory_time: float = 10.0,
    num_time_steps: int = 500,
) -> jnp.ndarray:
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
    bound_range = ub - lb

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

    def generate_chirp_targets(key: jax.Array) -> jnp.ndarray:
        # Feet Home Relative to Base:
        feet_home = jnp.array([
            [0.19215678, -0.142, -0.26637250],      # Front Right
            [0.19215678, 0.142, -0.26637250],       # Front Left
            [-0.19464322, -0.142, -0.26637250],     # Hind Right
            [-0.19464322, 0.142, -0.26637250],      # Hind Left
        ])

        # Raise Z Home Slightly:
        center_offset = jnp.array([0.0, 0.0, 0.08])
        feet_center = feet_home + center_offset

        # Assymetric Bias Limits:
        z_long = 0.20
        z_short = 0.10

        y_short = 0.05
        y_long = 0.15

        x_short = 0.15
        x_long = 0.30

        # Per-Foot Boundaries:
        fr_lb = feet_center[0] + jnp.array([-x_short, -y_long, -z_long])
        fr_ub = feet_center[0] + jnp.array([x_long, y_short, z_short])
        fl_lb = feet_center[1] + jnp.array([-x_short, -y_short, -z_long])
        fl_ub = feet_center[1] + jnp.array([x_long, y_long, z_short])

        rr_lb = feet_center[2] + jnp.array([-x_long,  -y_long, -z_long])
        rr_ub = feet_center[2] + jnp.array([x_short,  y_short,   z_short])
        rl_lb = feet_center[3] + jnp.array([-x_long,  -y_short,  -z_long])
        rl_ub = feet_center[3] + jnp.array([x_short,  y_long,  z_short])

        boundary_lb = jnp.stack([fr_lb, fl_lb, rr_lb, rl_lb])
        boundary_ub = jnp.stack([fr_ub, fl_ub, rr_ub, rl_ub])

        # Random Bias Offset:
        margin = 0.05
        bias_lb = boundary_lb + margin
        bias_ub = boundary_ub - margin

        key, bias_key = jax.random.split(key)
        bias_offset = jax.random.uniform(
            bias_key, shape=(4, 3), minval=bias_lb, maxval=bias_ub,
        )

        # Geometric Amplitude Limits:
        dist_to_ub = boundary_ub - bias_offset
        dist_to_lb = bias_offset - boundary_lb
        max_geom_amp = jnp.minimum(dist_to_ub, dist_to_lb)

        # Frequency and Phase Generation:
        key, frequency_key = jax.random.split(key)
        f_start = 0.1
        f_end = jax.random.uniform(frequency_key, minval=10.0, maxval=10.0)

        t = jnp.linspace(0, trajectory_time, num_time_steps)[:, None, None]
        freq_inst = f_start + (f_end - f_start) * (t / trajectory_time)
        k = (f_end - f_start) / trajectory_time
        chirp_phase = 2 * jnp.pi * (f_start * t + (k / 2) * t**2)

        key, phase_key = jax.random.split(key)
        phase_offsets = jax.random.uniform(
            phase_key, shape=(4, 3), minval=0.0, maxval=2 * jnp.pi
        )

        # Velocity Limits:
        key, velocity_key = jax.random.split(key)
        target_velocity = jax.random.uniform(
            velocity_key, shape=(4, 3), minval=0.5, maxval=2.0,
        )
        max_velocity_amp = target_velocity / (2 * jnp.pi * freq_inst)

        # Amplitude Variation and Scaling:
        key, scale_key, sign_key = jax.random.split(key, 3)
        amp_scale = jax.random.uniform(
            scale_key, shape=(4, 3), minval=0.1, maxval=1.0,
        )
        amp_sign = jax.random.choice(
            sign_key, jnp.array([-1.0, 1.0]), shape=(4, 3)
        )

        min_amp = 0.01
        allowed_amp = jnp.minimum(max_geom_amp[None, :, :], max_velocity_amp)
        final_amp = jnp.maximum(min_amp, allowed_amp * amp_scale)
        final_amp = jnp.minimum(final_amp, max_geom_amp[None, :, :])
        final_amp = final_amp * amp_sign[None, :, :]

        # Calculate Targets using the frequency-adjusted amplitude
        targets = bias_offset[None, :, :] + \
            final_amp * jnp.sin(chirp_phase + phase_offsets[None, :, :])

        return targets

    # Choose Target Generation Method
    key, target_key, bernoulli_key = jax.random.split(key, 3)
    step_targets = generate_step_targets(target_key, max_switches=max_switches, minimum_step_duration=minimum_step_duration)
    sinusoidal_targets = generate_sinusoidal_targets(target_key)
    chirp_targets = generate_chirp_targets(target_key)
    mask = jax.random.bernoulli(bernoulli_key, p=step_function_prob)
    targets = jnp.where(
        mask,
        step_targets,
        chirp_targets,
    )

    # Clip to Safe Zones:
    # targets = jnp.clip(targets, lb[None, ...], ub[None, ...])

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
        return dist_error + 0.001 * reg_error

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
        return q_solved, q_solved

    # Warm start and run scan:
    init_q = home_position
    final_q, q_trajectory = jax.lax.scan(solve_step, init_q, targets)

    return q_trajectory, targets


def generate_ik_trajectory(
    key: jax.Array,
    mj_model: mujoco.MjModel,
    trajectory_time: float = 10.0,
    num_time_steps: int = 500,
) -> jnp.ndarray:
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
    bound_range = ub - lb

    # --- 1. GENERATE JOINT SPACE CHIRPS ---
    def generate_joint_chirps(key):
        # We want to chirp around the HOME configuration
        # q_home shape: (12,)

        # Amplitudes for [Hip, Thigh, Calf]
        # Note: We give the Calf a LARGE amplitude to force movement
        # Hip: Small (0.2), Thigh: Med (0.3), Calf: Large (0.6)
        base_amps = jnp.tile(jnp.array([0.2, 0.3, 0.6]), 4)

        t = jnp.linspace(0, trajectory_time, num_time_steps)[:, None] # (T, 1)

        # Randomize Frequencies per joint (12 distinct frequencies)
        key, f_key = jax.random.split(key)
        f_start = 0.1
        f_end = jax.random.uniform(f_key, shape=(12,), minval=1.5, maxval=4.0)

        # Chirp Signal
        k = (f_end - f_start) / trajectory_time
        phase = 2 * jnp.pi * (f_start * t + (k / 2) * t**2)

        # Randomize Phase Offsets
        key, ph_key = jax.random.split(key)
        offsets = jax.random.uniform(ph_key, shape=(12,), minval=0, maxval=2*jnp.pi)

        # Randomize Amplitude Scaling
        key, amp_key = jax.random.split(key)
        amp_scales = jax.random.uniform(amp_key, shape=(12,), minval=0.5, maxval=1.0)

        # Calculate Desired Joint Trajectory
        # Shape: (T, 12)
        q_desired = home_position[None, :] + \
                    (base_amps * amp_scales)[None, :] * jnp.sin(phase + offsets[None, :])

        return q_desired

    # Choose Target Generation Method
    key, target_key = jax.random.split(key, 2)
    joint_targets = generate_joint_chirps(target_key)

    # --- 2. FORWARD KINEMATICS (Get Unsafe Foot Positions) ---
    # We need to map q_desired -> feet_positions
    # Scan FK over the trajectory
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

    # Vmap FK over time (T, 12) -> (T, 4, 3)
    feet_targets_unsafe = jax.vmap(get_fk)(joint_targets)

    targets_clamped = jnp.clip(
        feet_targets_unsafe,
        lb[None, ...],
        ub[None, ...],
    )

    # Loss Function: Distance to Target + Regularization
    def ik_loss(q, target_pos, target_qpos):
        current_pos = fk_fn(q)
        dist_error = jnp.sum((current_pos - target_pos) ** 2)
        reg_error = jnp.sum((q - target_qpos) ** 2)
        return dist_error + 0.1 * reg_error

    # Gradient Step
    grad_fn = jax.value_and_grad(ik_loss)

    def solve_step(carry, target):
        """
            Solves IK for a single timestep using an inner optimization loop.
            carry_q: The solution from the previous timestep (Warm start)
            target: The Cartesian target for the current timestep
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
        return q_solved, q_solved

    # Warm start and run scan:
    init_q = (home_position, home_position)
    final_q, q_trajectory = jax.lax.scan(solve_step, init_q, targets)

    return q_trajectory, targets
