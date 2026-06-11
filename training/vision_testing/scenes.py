import mujoco
import mujoco.viewer
from mujoco_scenes import generate


model = mujoco.MjModel.from_xml_path("../envs/unitree_go2/mjcf/scene_mjx_vendor_torque_steps.xml")




mujoco.viewer.launch(model)


