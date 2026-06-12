from absl import app, flags

import pathlib
import pickle

import mujoco

from regression.utilities.model_utilities import log_cholesky_to_mujoco
from training.envs.utilities.model_utilities import rehydrate_model


FLAGS = flags.FLAGS
flags.DEFINE_string('parameter_checkpoint', None, 'Path to the parameter file.')


def main(argv=None):
    parameter_checkpoint = pathlib.Path(FLAGS.parameter_checkpoint)
    if not parameter_checkpoint.is_dir():
        raise ValueError(f"Parameter checkpoint directory not found at {parameter_checkpoint}")

    with open(parameter_checkpoint / "checkpoint.pkl", 'rb') as f:
        checkpoint = pickle.load(f)

    with open(parameter_checkpoint / "config.pkl", 'rb') as f:
        config = pickle.load(f)

    print("Regressed Parameters:")
    for key, value in checkpoint['parameters'].items():
        if 'log_cholesky_inertia' in key:
            print(f"{key}: \n")
            for i, body_id in enumerate(checkpoint['spec']['log_cholesky_inertia']['body_ids']):
                body_mass, body_ipos, body_inertia, body_iquat = log_cholesky_to_mujoco(value[i])
                print(f"Body: {body_id}")
                print(f"\t{body_id}:\n")
                print(f"\t\tMass: {body_mass}\n")
                print(f"\t\tCenter of Mass: {body_ipos}\n")
                print(f"\t\tInertia: {body_inertia}\n")
                print(f"\t\tQuaternion: {body_iquat}\n")
        else:
            print(f"{key}: {value}")

    # Rehydrate Model from Parameter Checkpoint:
    model_parameters = {
        k: v
        for k, v in checkpoint['parameters'].items()
    }
    spec = checkpoint['spec']

    model_path = pathlib.Path("training/envs/unitree_go2_backflip/mjcf/scene_mjx_vendor_torque.xml")
    mj_model = mujoco.MjModel.from_xml_path(model_path.as_posix())

    mj_model = rehydrate_model(
        model=mj_model,
        parameters=model_parameters,
        regression_spec=spec
    )

    # Save Model to XML for Verification:
    output_xml_path = parameter_checkpoint / "rehydrated_model.xml"
    mujoco.mj_saveLastXML(str(output_xml_path), mj_model)
    print(f"Rehydrated model saved to {output_xml_path}")


if __name__ == '__main__':
    app.run(main)
