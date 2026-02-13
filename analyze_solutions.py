from absl import app
from pathlib import Path

import jax

from regression.tools.analyze_regression import analyze_regression


def main(argv=None):

    run_names = [
        "kind-pond-70",
        "wild-capybara-68",
        "swept-river-69",
        "distinctive-planet-71",
        "volcanic-yogurt-77",
        "dandy-leaf-75",
        "classic-water-76",
        "robust-energy-78",
        "toasty-universe-72",
        "different-river-73",
        "lively-water-79",
        "curious-wind-80",
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
