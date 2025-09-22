import os

from absl import app, flags

import onnx
from jax2onnx import to_onnx

from training.envs.unitree_go2.unitree_go2_joystick import UnitreeGo2Env
from training.algorithms.ppo.load_utilities import load_policy

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)


def main(argv=None):
    # Load Policy:
    env = UnitreeGo2Env()
    make_policy, params, metadata = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params, deterministic=True)

    # Convert to ONNX
    onnx_model = to_onnx(
        inference_function,
        input=[(1, env.observation_size["state"][0])],
        model_name=FLAGS.checkpoint_name,
        opset=11,
    )

    # Save the model
    output_path = f"onnx_models/{FLAGS.checkpoint_name}.onnx"
    onnx.save_model(onnx_model, output_path)


if __name__ == '__main__':
    app.run(main)
