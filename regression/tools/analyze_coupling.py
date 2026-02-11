from typing import Callable, Tuple, Dict

from absl import app, flags

from pathlib import Path
import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx

import plotly.express as px

from ..utilities import evaluation
from ..utilities.data_utilities import chunk_and_flatten_dataset, shuffle_data
from ..utilities.model_utilities import hydrate_model
from ..utilities.loss_utilities import loss_function
from ..utilities.evaluation import init_function, step_function
from ..utilities.typedefs import Dataset

jax.config.update('jax_enable_x64', True)

FLAGS = flags.FLAGS
flags.DEFINE_string('parameter_checkpoint', None, 'Path to the directory containing the regressed parameters and config.pkl', required=True)
flags.DEFINE_boolean('load_analysis', False, 'Whether to load existing analysis results')


def analyze_parameter_coupling(
    loss_fn: Callable, params: Dict[str, jax.Array], dataset: Dataset, batch_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
        Computes the Correlation Matrix of the parameters using the Hessian.
    """
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    def flat_loss_fn(theta, b):
        p = unflatten_fn(theta)
        return loss_fn(p, b)

    @jax.jit
    def accumulate_hessian(carry, batch):
        # Forward-over-Forward Hessian for MJX compatibility
        hessian_sum, loss_sum = carry
        hessian_batch = jax.jacfwd(jax.jacfwd(flat_loss_fn))(flat_params, batch)
        loss_batch = flat_loss_fn(flat_params, batch)
        hessian_sum = hessian_sum + hessian_batch
        loss_sum = loss_sum + loss_batch
        return (hessian_sum, loss_sum), None

    @jax.jit
    def _compute_kernel(hessian: jnp.ndarray, mse_value: float) -> Tuple[jnp.ndarray, jnp.ndarray]:        
        # For MSE loss, Covariance approx = 2 * sigma^2 * H^-1
        # we assume sigma^2 (noise variance) is approximated by the loss value itself:
        epsilon = 1e-6
        hessian_inv = jnp.linalg.inv(hessian + jnp.eye(hessian.shape[0]) * epsilon)

        # Covariance: (Approximation for Least Squares)
        sigma = 2.0 * mse_value * hessian_inv

        # Normalize (Correlation)
        diag = jnp.diag(sigma)
        std_devs = jnp.sqrt(jnp.maximum(diag, 1e-16))
        outer_std = jnp.outer(std_devs, std_devs)

        correlation = sigma / outer_std
        return correlation, hessian

    # Reshape (num_trials, time, joints) -> (num_batches, batch_size, time, joints)
    def reshape_to_batches(x):
        n_trials = x.shape[0]
        n_batches = n_trials // batch_size
        cutoff = n_batches * batch_size
        subset = x[:cutoff]
        return subset.reshape(n_batches, batch_size, *x.shape[1:])
    
    batched_dataset = jax.tree.map(reshape_to_batches, dataset)
    num_batches = jax.tree.leaves(batched_dataset)[0].shape[0]

    # Initialize Accumulators:
    hessian_init = jnp.zeros((flat_params.shape[0], flat_params.shape[0]))
    loss_init = 0.0

    (hessian_sum, loss_sum), _ = jax.lax.scan(accumulate_hessian, (hessian_init, loss_init), batched_dataset)

    hessian = hessian_sum / num_batches
    loss = loss_sum / num_batches
    
    correlation_matrix, hessian_matrix = _compute_kernel(
        hessian,
        loss,
    )

    return correlation_matrix, hessian_matrix


def main(argv=None):
    # Set up paths:
    package_root = Path(__file__).resolve().parent.parent

    # Load Regressed Parameters and Config:
    parameter_checkpoint_path = Path(FLAGS.parameter_checkpoint) / 'regressed_params.pkl'
    with open(parameter_checkpoint_path, 'rb') as f:
        params = pickle.load(f)

    config_path = Path(FLAGS.parameter_checkpoint) / 'config.pkl'
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Clean up params:
    params = {
        k: v
        for k, v in params.items()
        if not k.startswith('initial_')
    }

    if not FLAGS.load_analysis:
        # Load and Hydrate MuJoCo Model:
        mj_model_filepath = (package_root / "mjcf/scene_mjx_vendor.xml").resolve()
        mj_model = mujoco.MjModel.from_xml_path(str(mj_model_filepath))
        mj_model = hydrate_model(params, mj_model, config['regression'])

        # Create Static MJX Model:
        mjx_model_static = mjx.put_model(mj_model, impl="jax")
        n_substeps = int(config['physics']['control_rate'] / mj_model.opt.timestep)

        # Wrap Step Function:
        init_fn = evaluation.init_function
        step_fn = functools.partial(
            evaluation.step_function,
            n_substeps=n_substeps,
        )

        # Load Data:
        datasets = []
        dataset_directories = config['datasets'] if isinstance(config['datasets'], tuple) else (config['datasets'],)
        for directory_name in dataset_directories:
            data_path = Path(directory_name) / 'processed_data.pkl'
            if not data_path.exists():
                raise FileNotFoundError(f"{data_path} not found")

            print(f"Loading dataset: {directory_name}")
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)

            effective_window = config['training']['window_length'] + 1
            ds_chunked = chunk_and_flatten_dataset(
                jnp.array(data_dict['qpos']),
                jnp.array(data_dict['qvel']),
                jnp.array(data_dict['actuator_force']),
                jnp.array(data_dict['ctrl']),
                effective_window,
            )
            datasets.append(ds_chunked)

        dataset = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *datasets
        )

        # Loss Function: (MSE for Correlation Analysis)
        loss_fn = functools.partial(
            loss_function,
            model_static=mjx_model_static,
            init_function=init_fn,
            step_function=step_fn,
            objective_function=evaluation.get_objective_fn('mse'),
            objective_weights=config['loss']['weights'],
            regression_spec=config['regression'],
        )

        # Compute Correlation Matrix:
        batch_size = 32
        correlation_matrix, hessian_matrix = analyze_parameter_coupling(
            loss_fn,
            params,
            dataset,
            batch_size,
        )

        correlation_matrix = np.asarray(correlation_matrix)

        # Correlation Matrix Analysis:
        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
        condition_number = np.linalg.cond(correlation_matrix)

        print("Correlation Matrix Analysis:")
        print(f"Correlation Matrix Eigenvalues: {eigenvalues}")
        print(f"Correlation Matrix Condition Number: {condition_number}")

        # Hessian Analysis:
        hessian_eigenvalues, hessian_eigenvectors = np.linalg.eig(hessian_matrix)
        hessian_condition_number = np.linalg.cond(hessian_matrix)

        print("Hessian Matrix Analysis:")
        print(f"Hessian Matrix Eigenvalues: {hessian_eigenvalues}")
        print(f"Hessian Matrix Condition Number: {hessian_condition_number}")

        # Save Analysis Results:
        analysis_results = {
            "correlation_matrix": correlation_matrix,
            "correlation_matrix_eigenvalues": eigenvalues,
            "correlation_matrix_eigenvectors": eigenvectors,
            "correlation_matrix_condition_number": condition_number,
            "hessian_matrix": hessian_matrix,
            "hessian_matrix_eigenvalues": hessian_eigenvalues,
            "hessian_matrix_eigenvectors": hessian_eigenvectors,
            "hessian_matrix_condition_number": hessian_condition_number,
        }
        pickle_path = Path(FLAGS.parameter_checkpoint) / "coupling_analysis.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(analysis_results, f)

        # Plot Correlation Matrix:
        labels = []
        for key, value in sorted(params.items()):
            size = value.size
            if size == 1:
                labels.append(key)
            else:
                labels.extend([f"{key}_{i}" for i in range(size)])

        fig = px.imshow(
            correlation_matrix,
            x=labels,
            y=labels,
            zmin=-1, 
            zmax=1,
            color_continuous_scale='RdBu_r', 
            text_auto='.2f',
            aspect='equal',
            title="Parameter Correlation Matrix (Fisher Information)"
        )

        fig.update_layout(
            width=700,
            height=700,
            title_x=0.5,
            xaxis_title="Parameters",
            yaxis_title="Parameters",
        )
        
        html_path = Path(FLAGS.parameter_checkpoint) / "correlation_matrix.html"
        fig.write_html(str(html_path))
    else:
        # Load Existing Analysis Results:
        pickle_path = Path(FLAGS.parameter_checkpoint) / "coupling_analysis.pkl"
        if not pickle_path.exists():
            raise FileNotFoundError(f"{pickle_path} not found. Please run the analysis first without --load_analysis.")
        
        with open(pickle_path, 'rb') as f:
            analysis_results = pickle.load(f)

        correlation_matrix = analysis_results["correlation_matrix"]
        eigenvalues = analysis_results["correlation_matrix_eigenvalues"]
        condition_number = analysis_results["correlation_matrix_condition_number"]

        hessian_matrix = analysis_results["hessian_matrix"]
        hessian_eigenvalues = analysis_results["hessian_matrix_eigenvalues"]
        hessian_condition_number = analysis_results["hessian_matrix_condition_number"]

        with np.printoptions(precision=3, suppress=True, linewidth=100):
            print("Loaded Correlation Matrix Analysis:")
            print(f"Correlation Matrix Eigenvalues: {eigenvalues}")
            print(f"Correlation Matrix Minimum and Max Eigen Values: {np.min(eigenvalues)}, {np.max(eigenvalues)}")
            print(f"Correlation Matrix Condition Number: {condition_number}")

            print("Hessian Matrix Analysis:")
            print(f"Hessian Matrix Eigenvalues: {hessian_eigenvalues}")
            print(f"Hessian Matrix Minimum and Max Eigen Values: {np.min(hessian_eigenvalues)}, {np.max(hessian_eigenvalues)}")
            print(f"Hessian Matrix Condition Number: {hessian_condition_number}")


if __name__ == '__main__':
    app.run(main)
