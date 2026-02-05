from absl import app
import os
import mujoco
import numpy as np
import time

def main(argv=None):
    # Path to your model
    filename = 'mjcf/scene_mjx.xml'
    filepath = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(filepath):
        print(f"Error: Model file not found at {filepath}")
        return

    # Load Model and Data
    m = mujoco.MjModel.from_xml_path(filepath)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            # Lower Bound Z:
            # d.qpos[:3] = [0.3, 0.33, -0.83]
            # # Upper Bound Z:
            # d.qpos[:3] = [-0.13, 3.45, -0.83]
            # # Upper Bound X:
            # d.qpos[:3] = [-0.13, -1.1, -0.83]
            # # Bound Y:
            # d.qpos[:3] = [-1.05, 0.33, -0.83]

            mujoco.mj_forward(m, d)

            site_position = d.site("front_right_foot").xpos
            print(f"Site Pos: {site_position}")

            # Sync viewer with the new physics state
            viewer.sync()
            time.sleep(0.01)

    # 1. Set Robot to Home Position
    # Check if 'home' keyframe exists, otherwise use default qpos
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key_id != -1:
        print("Found 'home' keyframe. applying...")
        d.qpos = m.keyframe('home').qpos
    else:
        print("No 'home' keyframe found. Using default qpos (0).")
        # Ensure qpos is set to default (usually 0 or nominal)
        mujoco.mj_resetDataKeyframe(m, d, 0) 

    # 2. Forward Kinematics
    mujoco.mj_forward(m, d)

    # 3. Get IDs
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    feet_names = [
        'front_right_foot',
        'front_left_foot',
        'hind_right_foot',
        'hind_left_foot',
    ]

    print("\n" + "="*60)
    print(f"{'Foot Name':<20} | {'World Position (X, Y, Z)':<30}")
    print("-" * 60)

    # Base Position and Rotation matrix
    base_pos = d.xpos[base_id]
    base_mat = d.xmat[base_id].reshape(3, 3)

    feet_relative_positions = []

    for name in feet_names:
        site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)

        if site_id == -1:
            print(f"{name:<20} | NOT FOUND")
            continue

        # Get World Position
        foot_world_pos = d.site_xpos[site_id]

        # Calculate Relative Position: R.T * (P_foot - P_base)
        rel_pos = base_mat.T @ (foot_world_pos - base_pos)
        feet_relative_positions.append(rel_pos)

        print(f"{name:<20} | {foot_world_pos}")

    print("\n" + "="*60)
    print("COPY PASTE THIS INTO YOUR SCRIPT (feet_home):")
    print("-" * 60)
    
    print("feet_home = jnp.array([")
    for i, pos in enumerate(feet_relative_positions):
        # formatted for code
        print(f"    [{pos[0]:.8f}, {pos[1]:.8f}, {pos[2]:.8f}],  # {feet_names[i]}")
    print("])")
    print("="*60 + "\n")

if __name__ == '__main__':
    app.run(main)