from absl import app, flags

import pickle
import pathlib
import pprint


FLAGS = flags.FLAGS
flags.DEFINE_string('parameter_checkpoint', None, 'Path to the pickle file containing the parameters', short_name='p', required=True)


def main(argv=None):
    parameter_path = pathlib.Path(FLAGS.parameter_checkpoint)

    if not parameter_path.exists():
        raise FileNotFoundError(f"Parameter checkpoint not found at {parameter_path}")

    pickle_path = parameter_path / "regressed_params.pkl"

    with open(pickle_path, 'rb') as f:
        parameters = pickle.load(f)

    # Formate the parameters for better readability:
    leg_mapping = {
        'front_right': [0, 1, 2],
        'front_left': [3, 4, 5],
        'rear_right': [6, 7, 8],
        'rear_left': [9, 10, 11]
    }
    output = {leg: {} for leg in leg_mapping.keys()}
    for key, value in parameters.items():
        if key.startswith('initial'): continue
        for leg, indices in leg_mapping.items():
            output[leg][key] = value[indices]

    print("Regressed Parameters:")
    pprint.pprint(output)


if __name__ == '__main__':
    app.run(main)
