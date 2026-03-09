import mujoco
import numpy as np

def calculate_go2_radius(xml_path: str):
    # Load the model
    # Note: Ensure your mesh files are correctly located in '../assets' 
    # as defined in your XML, or MuJoCo will throw a load error.
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except ValueError as e:
        print(f"Error loading model: {e}")
        print("Tip: If you are missing the .obj files, you can temporarily remove the <asset> and <geom mesh=...> tags to calculate the kinematic radius.")
        return

    data = mujoco.MjData(model)

    # 1. Evaluate forward kinematics at the default joint configuration (data.qpos = 0)
    qpos = np.array(model.keyframe('home').qpos)
    data.qpos = qpos
    mujoco.mj_kinematics(model, data)

    # 2. Extract the global XY position of the base
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_xy = data.xpos[base_id][:2]

    # 3. Define the foot sites as named in your XML
    foot_site_names = [
        "front_right_foot",
        "front_left_foot",
        "hind_right_foot",
        "hind_left_foot"
    ]

    radii = []
    print("--- Individual Foot Radii ---")
    for name in foot_site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        
        if site_id == -1:
            print(f"Warning: Site '{name}' not found in model.")
            continue
            
        # Extract the global XY position of the foot site
        foot_xy = data.site_xpos[site_id][:2]
        
        # Calculate the Euclidean distance in the XY plane
        radius = np.linalg.norm(foot_xy - base_xy)
        radii.append(radius)
        print(f"{name:<20}: {radius:.4f} m")

    # 4. Calculate the characteristic average radius
    if radii:
        r_char = np.mean(radii)
        print("\n==================================")
        print(f"Characteristic Radius (R_char): {r_char:.4f} m")
        print("==================================")
        return r_char

if __name__ == "__main__":
    # Replace with the path to your Go2 XML file
    calculate_go2_radius("training/envs/unitree_go2_minimal/mjcf/scene_mjx_vendor_position.xml")