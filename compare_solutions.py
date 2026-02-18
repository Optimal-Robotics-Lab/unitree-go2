from absl import app, flags

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


flags.DEFINE_string('folder', None, 'Path to the folder containing .pkl files', required=True)


def submatrix_analysis(matrix: np.ndarray, num_param_types: int, param_names: list = None):
    """
        Args:
            matrix: An (N, N) matrix.
            num_param_types: The number of parameter types
            param_names: Optional list of parameter type names for labeling.
    """
    N_JOINTS = 12

    expected_size = N_JOINTS * num_param_types
    if matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match expected "
            f"size ({expected_size}, {expected_size}) for {num_param_types} parameters."
        )

    leg_definitions = {
        "Front Right (FR)": {"base_idx": [0, 1, 2],   "pos": (1, 1)},
        "Front Left (FL)":  {"base_idx": [3, 4, 5],   "pos": (1, 2)},
        "Back Right (BR)":  {"base_idx": [6, 7, 8],   "pos": (2, 1)},
        "Back Left (BL)":   {"base_idx": [9, 10, 11], "pos": (2, 2)}
    }

    if param_names is None or len(param_names) != num_param_types:
        param_names = [f"Param {i+1}" for i in range(num_param_types)]

    axis_labels = []
    for p_name in param_names:
        for j in range(3):
            axis_labels.append(f"{p_name}<br>J{j}")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(leg_definitions.keys()),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    leg_results = {}

    for leg_name, leg_info in leg_definitions.items():
        base_idx = leg_info["base_idx"]
        row, col = leg_info["pos"]
        leg_indices = []

        for block_idx in range(num_param_types):
            offset = block_idx * N_JOINTS
            leg_indices.extend([i + offset for i in base_idx])

        submatrix = matrix[np.ix_(leg_indices, leg_indices)]

        eigenvalues, eigenvectors = np.linalg.eig(submatrix)
        cond_num = np.linalg.cond(submatrix)

        leg_results[leg_name] = {
            "matrix": submatrix,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "condition_number": cond_num,
            "indices": leg_indices
        }

        fig.layout.annotations[len(leg_results) - 1].text = f"{leg_name}<br>Cond: {cond_num:.2f}"

        # Add the Heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=submatrix,
                x=axis_labels,
                y=axis_labels,
                zmin=-1.0,
                zmax=1.0,
                colorscale='RdBu',
                showscale=(row == 1 and col == 2),
                colorbar=dict(title="Correlation") if (row == 1 and col == 2) else None
            ),
            row=row, col=col
        )

        fig.update_xaxes(scaleanchor=f"y{len(leg_results)}", scaleratio=1, row=row, col=col)
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    # Final layout updates
    fig.update_layout(
        title_text=f"Intra-Leg Parameter Correlation Matrices ({num_param_types} Param Types)",
        height=800,
        width=900,
        plot_bgcolor='white'
    )

    fig.show()

    return leg_results


def main(argv=None):
    folder_path = Path(flags.FLAGS.folder)
    if not folder_path.exists():
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    data_list = []
    target_filename = 'optimization_analysis.pkl'

    for path in folder_path.rglob(target_filename):
        print(f"Processing: {path}")
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                data['name'] = path.parent.name
                data_list.append(data)
            else:
                print(f"Warning: {path} did not contain a dictionary.")

        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not data_list:
        print("No valid data found to export.")
        return

    # Create DataFrame
    df = pd.DataFrame(data_list)

    # Remove Matrix and Eigenvector columns:
    cols_to_drop = [
        "correlation_matrix", "correlation_matrix_eigenvectors",
        "relative_hessian_matrix", "relative_hessian_matrix_eigenvectors",
        "hessian_matrix", "hessian_matrix_eigenvectors"
    ]
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns containing '_matrix'...")
        df = df.drop(columns=cols_to_drop)

    # Sort by the number of labels in the 'labels' column, if it exists:
    if 'param_labels' in df.columns:
        print("Sorting rows by the number of labels...")
        df = df.sort_values(
            by='param_labels',
            key=lambda col: col.apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0),
            ascending=False
        )
    else:
        print("Warning: 'param_labels' column not found. Skipping sort.")

    # Reorder columns to put 'name' first, if it exists:
    if 'name' in df.columns:
        cols = ['name'] + [c for c in df.columns if c != 'name']
        df = df[cols]

    # Show as HTML table
    output_file = 'comparison.html'
    df.to_html(output_file, index=False)


if __name__ == "__main__":
    app.run(main)
