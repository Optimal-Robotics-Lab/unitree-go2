
def step_function(env: unitree_go2_joystick.UnitreeGo2Env, data: mujoco.MjData, action: npt.NDArray, n_substeps: int) -> mujoco.MjData:
    # Set Motor Model:
    if env.motor_config is not None:
        def motor_model(mj_data: mujoco.MjData, target_qpos: npt.NDArray) -> npt.NDArray:
            # Extract Joint States:
            joint_positions = mj_data.qpos[7:]
            joint_velocities = mj_data.qvel[6:]

            # PD Control Law
            desired_torque = env.motor_config.kp * (target_qpos - joint_positions) \
                - env.motor_config.kv * joint_velocities

            # Torque Speed Curve:
            available_torque = env.motor_config.tau_max - (env.motor_config.damping_slope * np.abs(joint_velocities))
            available_torque = np.maximum(available_torque, 0.0)

            # Apply Torque Limits:
            torque = np.clip(desired_torque, -available_torque, available_torque)

            return torque
    else:
        def motor_model(mj_data: mujoco.MjData, target_qpos: npt.NDArray) -> npt.NDArray:
            return target_qpos

    # Utility Functions:
    def simulation_step(env: unitree_go2_joystick.UnitreeGo2Env, data: mujoco.MjData, action: npt.NDArray, n_substeps: int) -> mujoco.MjData:
        # Compute Target Joint Positions from Action:
        target_qpos = env.default_pose + action * env.action_scale
        target_qpos = np.clip(target_qpos, env.joint_lb, env.joint_ub)

        # Run Physics Substeps:
        for _ in range(n_substeps):
            ctrl = motor_model(data, target_qpos)
            data.ctrl = ctrl
            mujoco.mj_step(env._mj_model, data)

        return data

# This is what we include
functools.partial(step_function, env, n_substeps=n_substeps)


def inference_wrapper(observation: jax.Array) -> jax.Array:
    dummy_key = jax.random.key(0)

    actions, info = model.get_actions(
        observation,
        dummy_key,
        deterministic=True
    )

    return actions