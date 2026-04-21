from absl import app, flags

from regression.tools.analyze_regression import analyze_regression

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'parameter_checkpoint',
    None,
    'Path to the parameter checkpoint directory containing regressed_params.pkl and config.pkl.',
    required=True,
    short_name='p',
)

flags.DEFINE_bool(
    'load_analysis',
    False,
    'Whether to load existing analysis results from optimization_analysis.pkl instead of recomputing.',
    required=False,
    short_name='l',
)


def main(argv=None):
    analyze_regression(FLAGS.parameter_checkpoint, FLAGS.load_analysis)


if __name__ == "__main__":
    app.run(main)
