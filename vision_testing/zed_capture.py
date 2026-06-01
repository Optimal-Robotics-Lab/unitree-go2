"""
Render the mounted ZED stereo pair (zedm_left_lens / zedm_right_lens) on the Go2
and emphasize GROUND-TRUTH DEPTH. 
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCENE = os.path.join(
    REPO, "training/envs/unitree_go2/mjcf/scene_mjx_vendor_torque_rough.xml"
)
OUT = HERE


W, H = 1280, 720
CAMS = ["zedm_left_lens", "zedm_right_lens"]
SKY = 10.0 

model = mujoco.MjModel.from_xml_path(SCENE)
model.vis.global_.offwidth = W
model.vis.global_.offheight = H

data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=H, width=W)
frames = {}
for cam in CAMS:
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    frames[cam] = {"rgb": rgb, "depth": depth}
    np.save(os.path.join(OUT, f"{cam}_depth.npy"), depth)

ground = np.concatenate([f["depth"][f["depth"] < SKY].ravel() for f in frames.values()])
vmin = float(np.percentile(ground, 1))
vmax = float(np.percentile(ground, 99))
print(f"ground depth (both eyes): {ground.min():.2f}..{ground.max():.2f} m  "
      f"| color scale {vmin:.2f}..{vmax:.2f} m  | median {np.median(ground):.2f} m")

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

depth_axes = []
for col, cam in enumerate(CAMS):
    d = frames[cam]["depth"].copy()
    d_show = np.where(d < SKY, d, np.nan)
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(d_show, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(f"{cam}  —  ground-truth depth", fontsize=12)
    ax.axis("off")
    depth_axes.append(ax)

cbar = fig.colorbar(im, ax=depth_axes, fraction=0.046, pad=0.02)
cbar.set_label("depth (meters)")

for col, cam in enumerate(CAMS):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(frames[cam]["rgb"])
    ax.set_title(f"{cam}  RGB (reference)", fontsize=9)
    ax.axis("off")

panel_path = os.path.join(OUT, "zed_depth_panel.png")
fig.savefig(panel_path, dpi=120, bbox_inches="tight")

for cam in CAMS:
    d = np.clip(frames[cam]["depth"], vmin, vmax)
    norm = ((d - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(OUT, f"{cam}_depth_vis.png"), colored)

print(f"wrote {panel_path}")
print(f"wrote per-lens *_depth_vis.png and raw *_depth.npy in {OUT}")
