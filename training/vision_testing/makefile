ASSETS_DIR := ../../training/envs/unitree_go2/assets

generate_steps:
		python3 -m mujoco_scenes.hfield steps --size 256 256 --step-size 16 --output $(ASSETS_DIR)/steps_hfield.png

generate_rough:
		python3 -m mujoco_scenes.hfield rough --size 256 256 --output $(ASSETS_DIR)/rough_hfield.png

generate_sin:
		python3 -m mujoco_scenes.hfield sine --size 256 256 --amplitude 0.1 --freq 2.0 --output $(ASSETS_DIR)/sine_hfield.png

# Tunable stair parameters (physical scale set in scene XML hfield size attr)
N_STAIRS := 10

generate_stairs:
		python3 -c "\
		import numpy as np; from PIL import Image; \
		n=$(N_STAIRS); size=256; band=size//n; \
		arr=np.zeros((size,size),dtype=np.uint8); \
		[arr.__setitem__((slice(i*band,(i+1)*band),slice(None)),int(i*255/(n-1))) for i in range(n)]; \
		Image.fromarray(arr).save('$(ASSETS_DIR)/stairs_hfield.png'); \
		print(f'saved stairs_hfield.png ({n} stairs)')"
