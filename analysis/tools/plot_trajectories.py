from absl import app, flags

import sys
from pathlib import Path

import numpy as np

import plotly.graph_objects as go
import plotly.express as px


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Base directory containing the processed runs.', short_name='d', required=True
)


# POLICY_NAME_MAP = {
#     "regressed-position-forward": "Regressed Parameter",
#     "regressed-position-dr-forward": "Regressed Parameter (DR)",
#     "transparent-position-forward": "Transparent Parameter",
#     "transparent-position-dr-forward": "Transparent Parameter (DR)",
#     "uniform-position-dr-forward": "Uniform Parameter (DR)",
#     "vendor-position-forward": "Vendor Baseline",
#     "vendor-position-dr-forward": "Vendor Baseline (DR)",
# }

POLICY_NAME_MAP = {
    "fresh-armadillo-6": "Regressed Parameter",
    "rose-fog-12": "Regressed Parameter w/ Domain Randomization",
    "treasured-sky-23": "Transparent Parameter",
    "resilient-oath-10": "Transparent Parameter w/ Domain Randomization",
    "major-bush-14": "Uniform Parameter Domain Randomization",
    "astral-bee-25": "Vendor",
    "lilac-resonance-8": "Vendor w/ Domain Randomization",
}


def load_trajectory_data(base_dir: Path):
    """Loads and stacks Vicon trajectory data by system."""
    system_data = {}

    # trajectory_file = "postprocessed_filtered_vicon_history.csv"
    trajectory_file = "trajectory.csv"

    for csv_path in base_dir.rglob(trajectory_file):
        run_name = csv_path.parent.name
        system_name = run_name.rsplit('-', 1)[0]

        try:
            data = np.loadtxt(csv_path, delimiter=',')
            if data.size == 0:
                continue

            # Extract X and Y position
            x_pos = data[:, 1]
            y_pos = data[:, 2]

            if system_name not in system_data:
                system_data[system_name] = {'x': [], 'y': []}

            system_data[system_name]['x'].append(x_pos)
            system_data[system_name]['y'].append(y_pos)

        except Exception as e:
            print(f"Error loading {csv_path}: {e}")

    # Convert lists to 2D numpy arrays since all runs are the exact same length
    for sys_name in system_data:
        system_data[sys_name]['x'] = np.vstack(system_data[sys_name]['x'])
        system_data[sys_name]['y'] = np.vstack(system_data[sys_name]['y'])

    # Sort alphabetically so colors remain consistent across plots
    return dict(sorted(system_data.items()))


def hex_to_rgba(hex_color, alpha):
    """Helper to convert hex colors to RGBA for Plotly fills."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def plot_all_trajectories(system_data, output_dir: Path):
    """Plots every individual run as a thin line."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, (raw_sys_name, data_arrays) in enumerate(system_data.items()):
        clean_name = POLICY_NAME_MAP.get(raw_sys_name, raw_sys_name)
        color = colors[idx % len(colors)]

        x_runs = data_arrays['x']
        y_runs = data_arrays['y']
        num_runs = x_runs.shape[0]

        for i in range(num_runs):
            # Only add the label to the legend for the first run of the system
            show_legend = True if i == 0 else False

            fig.add_trace(go.Scatter(
                x=x_runs[i],
                y=y_runs[i],
                mode='lines',
                name=clean_name,
                legendgroup=clean_name,
                line=dict(color=color, width=1.5),
                opacity=0.4,
                showlegend=show_legend
            ))

    fig.update_layout(
        title=dict(text="Top-Down View: All Hardware Trajectories", font=dict(size=18)),
        xaxis_title="Forward Position (X) [m]",
        yaxis_title="Lateral Position (Y) [m]",
        template="simple_white",
        width=1000,
        height=800,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        )
    )

    # Add center line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

    fig.write_html(output_dir / "trajectories_all_runs.html")
    fig.write_image(output_dir / "trajectories_all_runs.pdf")
    print("Saved: trajectories_all_runs (HTML/PDF)")


def plot_statistical_trajectories(system_data, output_dir: Path):
    """Plots the mean trajectory with a shaded standard deviation region."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, (raw_sys_name, data_arrays) in enumerate(system_data.items()):
        clean_name = POLICY_NAME_MAP.get(raw_sys_name, raw_sys_name)
        hex_color = colors[idx % len(colors)]
        fill_color = hex_to_rgba(hex_color, 0.2)

        # Calculate statistics across the runs (axis 0)
        mean_x = np.mean(data_arrays['x'], axis=0)
        mean_y = np.mean(data_arrays['y'], axis=0)
        std_y = np.std(data_arrays['y'], axis=0)

        upper_bound = mean_y + std_y
        lower_bound = mean_y - std_y

        # 1. Upper Bound (Invisible line)
        fig.add_trace(go.Scatter(
            x=mean_x, y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            legendgroup=clean_name,
            hoverinfo='skip'
        ))

        # 2. Lower Bound (Invisible line, fills area to the Upper Bound)
        fig.add_trace(go.Scatter(
            x=mean_x, y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=fill_color,
            showlegend=False,
            legendgroup=clean_name,
            hoverinfo='skip'
        ))

        # 3. Mean Trajectory Line
        fig.add_trace(go.Scatter(
            x=mean_x, y=mean_y,
            mode='lines',
            name=clean_name,
            legendgroup=clean_name,
            line=dict(color=hex_color, width=3)
        ))

    fig.update_layout(
        title=dict(text="Statistical Average Trajectories with Lateral Variance (±1σ)", font=dict(size=18)),
        xaxis_title="Forward Position (X) [m]",
        yaxis_title="Lateral Position (Y) [m]",
        template="simple_white",
        width=1000,
        height=800,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        )
    )

    # Add center line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

    fig.write_html(output_dir / "trajectories_statistical.html")
    fig.write_image(output_dir / "trajectories_statistical.pdf")
    print("Saved: trajectories_statistical (HTML/PDF)")


def main(argv=None):
    base_dir = Path(FLAGS.directory_name).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: Directory {base_dir} does not exist.", file=sys.stderr)
        return

    print(f"Scanning for Vicon data in: {base_dir}")
    system_data = load_trajectory_data(base_dir)

    if not system_data:
        print("No 'postprocessed_filtered_vicon_history.csv' files found.", file=sys.stderr)
        return

    print("Generating trajectory plots...")
    plot_all_trajectories(system_data, base_dir)
    plot_statistical_trajectories(system_data, base_dir)

    print("\nPlotting complete.")


if __name__ == '__main__':
    app.run(main)
