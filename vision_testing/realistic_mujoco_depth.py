import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

W = 720
H = 1280

model = mujoco.MjModel.from_xml_path(os.path.join(REPO, "training/envs/unitree_go2/mjcf/scene_mjx_vendor_torque_rough.xml"))
data = mujoco.MjData(model)
 

def get_intrinsics(model: mujoco.MjModel, cam_name: str, W: int, H: int):
    cam = model.cam(cam_name)
    fov = float(cam.fovy[0])
    fy = (H / 2) / np.tan(np.radians(fov / 2))
    fx = fy  
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


print(get_intrinsics(model, "zedm_left_lens", W, H))