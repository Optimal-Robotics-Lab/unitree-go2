from absl import app
from pathlib import Path

import jax

from regression.tools.analyze_regression import analyze_regression


def main(argv=None):

    run_names = [
        "hardy-sound-81",
        "polar-moon-74",
    ]

    checkpoint_path = Path("regression/checkpoints")
    for i, run_name in enumerate(run_names):
        run_path = checkpoint_path / run_name

        # Verify that these runs exist:
        if not run_path.exists():
            raise ValueError(f"Run path does not exist: {run_path}")

        # Clear Cache between analyses:
        jax.clear_caches()

        # Run analysis:
        print(f"Analyzing run {i}: {run_name}")
        analyze_regression(run_path)



if __name__ == '__main__':
    app.run(main)
