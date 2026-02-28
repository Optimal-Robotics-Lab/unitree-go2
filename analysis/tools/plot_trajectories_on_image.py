from absl import app, flags

import sys
from pathlib import Path

import numpy as np
import cv2

import plotly.graph_objects as go
import plotly.express as px


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'directory_name', None, 'Base directory containing the processed runs.', short_name='d', required=True
)
flags.DEFINE_string(
    'camera_image', None, 'Path to the camera image for perspective projection.', short_name='c', required=False
)

POLICY_NAME_MAP = {
    "regressed-position-forward": "Regressed Parameter",
    "regressed-position-dr-forward": "Regressed Parameter (DR)",
    "transparent-position-forward": "Transparent Parameter",
    "transparent-position-dr-forward": "Transparent Parameter (DR)",
    "uniform-position-dr-forward": "Uniform Parameter (DR)",
    "vendor-position-forward": "Vendor Baseline",
    "vendor-position-dr-forward": "Vendor Baseline (DR)",
}


def load_trajectory_data(base_dir: Path):
    """Loads and stacks Vicon trajectory data by system."""
    system_data = {}

    for csv_path in base_dir.rglob("postprocessed_filtered_vicon_history.csv"):
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


def hex_to_bgr(hex_color):
    """Helper to convert hex colors to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r) # OpenCV uses BGR!


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
            show_legend = True if i == 0 else False
            fig.add_trace(go.Scatter(
                x=x_runs[i], y=y_runs[i],
                mode='lines', name=clean_name, legendgroup=clean_name,
                line=dict(color=color, width=1.5), opacity=0.4, showlegend=show_legend
            ))

    fig.update_layout(
        title=dict(text="Top-Down View: All Hardware Trajectories", font=dict(size=18)),
        xaxis_title="Forward Position (X) [m]", yaxis_title="Lateral Position (Y) [m]",
        template="simple_white", width=1000, height=800,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
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

        mean_x = np.mean(data_arrays['x'], axis=0)
        mean_y = np.mean(data_arrays['y'], axis=0)
        std_y = np.std(data_arrays['y'], axis=0)

        upper_bound = mean_y + std_y
        lower_bound = mean_y - std_y

        fig.add_trace(go.Scatter(
            x=mean_x, y=upper_bound, mode='lines', line=dict(width=0),
            showlegend=False, legendgroup=clean_name, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=mean_x, y=lower_bound, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=fill_color, showlegend=False,
            legendgroup=clean_name, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=mean_x, y=mean_y, mode='lines', name=clean_name,
            legendgroup=clean_name, line=dict(color=hex_color, width=3)
        ))

    fig.update_layout(
        title=dict(text="Statistical Average Trajectories with Lateral Variance (±1σ)", font=dict(size=18)),
        xaxis_title="Forward Position (X) [m]", yaxis_title="Lateral Position (Y) [m]",
        template="simple_white", width=1000, height=800,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

    fig.write_html(output_dir / "trajectories_statistical.html")
    fig.write_image(output_dir / "trajectories_statistical.pdf")
    print("Saved: trajectories_statistical (HTML/PDF)")


# def project_trajectories_to_camera(system_data, output_dir: Path, camera_img_path: str):
#     """Projects Vicon data onto a 2D camera image using a homography matrix."""
#     camera_img = cv2.imread(camera_img_path)
#     if camera_img is None:
#         print(f"Error: Could not load camera image from {camera_img_path}")
#         return

#     # Coordinates are relative to image:

#     # Vicon Coordinates relative to origin:
#     pts_vicon = np.float32([
#         [0.2,  2.0],    # Top Left
#         [10.1, 2.0],    # Top Right
#         [10.1,-2.0],    # Bottom Right
#         [0.2, -2.0]     # Bottom Left
#     ])

#     # Image Pixel Coordinates:
#     pts_camera = np.float32([
#         [1,    835],    # Top Left Pixel Coord
#         [1656, 1519],   # Top Right Pixel Coord
#         [1668, 2132],   # Bottom Right Pixel Coord
#         [1,    2734]    # Bottom Left Pixel Coord
#     ])

#     # Compute transformation matrix
#     matrix = cv2.getPerspectiveTransform(pts_vicon, pts_camera)
#     colors = px.colors.qualitative.Plotly

#     # Draw projected lines and shaded areas
#     for idx, (raw_sys_name, data_arrays) in enumerate(system_data.items()):
#         bgr_color = hex_to_bgr(colors[idx % len(colors)])

#         # Calculate statistics
#         mean_x = np.mean(data_arrays['x'], axis=0)
#         mean_y = np.mean(data_arrays['y'], axis=0)
#         std_y = np.std(data_arrays['y'], axis=0)

#         upper_bound = mean_y + std_y
#         lower_bound = mean_y - std_y

#         # Construct the closed polygon: forward along upper bound, backward along lower bound
#         upper_pts = np.vstack((mean_x, upper_bound)).T
#         lower_pts = np.vstack((mean_x, lower_bound)).T
#         polygon_pts = np.vstack((upper_pts, lower_pts[::-1]))

#         # Reshape for OpenCV and apply perspective transform
#         polygon_pts = polygon_pts.reshape(-1, 1, 2).astype(np.float32)
#         projected_polygon = cv2.perspectiveTransform(polygon_pts, matrix)
#         projected_polygon_int = np.int32(projected_polygon)

#         # Create an overlay for semi-transparency
#         overlay = camera_img.copy()
#         cv2.fillPoly(overlay, [projected_polygon_int], color=bgr_color)

#         # Blend the overlay with the original image (20% opacity)
#         alpha = 0.2
#         cv2.addWeighted(overlay, alpha, camera_img, 1 - alpha, 0, camera_img)

#         # Stack X and Y arrays into the shape OpenCV expects
#         vicon_pts = np.vstack((mean_x, mean_y)).T
#         vicon_pts = vicon_pts.reshape(-1, 1, 2).astype(np.float32)

#         # Apply transformation
#         projected_mean = cv2.perspectiveTransform(vicon_pts, matrix)
#         projected_mean_int = np.int32(projected_mean)

#         # Draw mean line directly on the camera_img on top of the shaded area
#         cv2.polylines(camera_img, [projected_mean_int], isClosed=False, color=bgr_color, thickness=4, lineType=cv2.LINE_AA)

#     output_path = output_dir / "projected_camera_view.jpg"
#     cv2.imwrite(str(output_path), camera_img)
#     print(f"Saved: {output_path}")


# def project_trajectories_to_camera(system_data, output_dir: Path, camera_img_path: str):
#     """Projects Vicon data onto a 2D camera image and saves both a JPG and a transparent vector PDF."""
#     camera_img = cv2.imread(camera_img_path)
#     if camera_img is None:
#         print(f"Error: Could not load camera image from {camera_img_path}")
#         return

#     h_cam, w_cam = camera_img.shape[:2]

#     # Coordinates are relative to the image:
#     # Vicon Coordinates relative to origin:
#     pts_vicon = np.float32([
#         [0.2,  2.0],    # Top Left
#         [10.1, 2.0],    # Top Right
#         [10.1,-2.0],    # Bottom Right
#         [0.2, -2.0]     # Bottom Left
#     ])

#     # Image Pixel Coordinates:
#     pts_camera = np.float32([
#         [1,    835],    # Top Left Pixel Coord
#         [1656, 1519],   # Top Right Pixel Coord
#         [1668, 2132],   # Bottom Right Pixel Coord
#         [1,    2734]    # Bottom Left Pixel Coord
#     ])

#     # Compute transformation matrix
#     matrix = cv2.getPerspectiveTransform(pts_vicon, pts_camera)
#     colors = px.colors.qualitative.Plotly

#     # Initialize a Plotly figure for the transparent vector PDF
#     fig_pdf = go.Figure()

#     # Draw projected lines and shaded areas
#     for idx, (raw_sys_name, data_arrays) in enumerate(system_data.items()):
#         hex_color = colors[idx % len(colors)]
#         bgr_color = hex_to_bgr(hex_color)
#         fill_color = hex_to_rgba(hex_color, 0.2)
#         clean_name = POLICY_NAME_MAP.get(raw_sys_name, raw_sys_name)

#         # Calculate statistics
#         mean_x = np.mean(data_arrays['x'], axis=0)
#         mean_y = np.mean(data_arrays['y'], axis=0)
#         std_y = np.std(data_arrays['y'], axis=0)
        
#         upper_bound = mean_y + std_y
#         lower_bound = mean_y - std_y

#         # --- 1. SHADED VARIANCE REGION ---
#         upper_pts = np.vstack((mean_x, upper_bound)).T
#         lower_pts = np.vstack((mean_x, lower_bound)).T
#         polygon_pts = np.vstack((upper_pts, lower_pts[::-1])) 
        
#         polygon_pts = polygon_pts.reshape(-1, 1, 2).astype(np.float32)
#         projected_polygon = cv2.perspectiveTransform(polygon_pts, matrix)
        
#         # Draw on OpenCV JPG
#         projected_polygon_int = np.int32(projected_polygon)
#         overlay = camera_img.copy()
#         cv2.fillPoly(overlay, [projected_polygon_int], color=bgr_color)
#         alpha = 0.2
#         cv2.addWeighted(overlay, alpha, camera_img, 1 - alpha, 0, camera_img)

#         # Add to Plotly Vector PDF
#         fig_pdf.add_trace(go.Scatter(
#             x=projected_polygon[:, 0, 0], y=projected_polygon[:, 0, 1], 
#             mode='lines', line=dict(width=0), fill='toself', 
#             fillcolor=fill_color, showlegend=False, hoverinfo='skip'
#         ))

#         # --- 2. MEAN TRAJECTORY LINE ---
#         vicon_pts = np.vstack((mean_x, mean_y)).T
#         vicon_pts = vicon_pts.reshape(-1, 1, 2).astype(np.float32)
#         projected_mean = cv2.perspectiveTransform(vicon_pts, matrix)
        
#         # Draw on OpenCV JPG
#         projected_mean_int = np.int32(projected_mean)
#         cv2.polylines(camera_img, [projected_mean_int], isClosed=False, color=bgr_color, thickness=4, lineType=cv2.LINE_AA)

#         # Add to Plotly Vector PDF
#         fig_pdf.add_trace(go.Scatter(
#             x=projected_mean[:, 0, 0], y=projected_mean[:, 0, 1], 
#             mode='lines', name=clean_name, line=dict(color=hex_color, width=3)
#         ))

#     # Save OpenCV Composite JPG
#     img_output_path = output_dir / "projected_camera_view.jpg"
#     cv2.imwrite(str(img_output_path), camera_img)
#     print(f"Saved: {img_output_path}")

#     # Configure and Save Transparent Vector PDF
#     fig_pdf.update_layout(
#         width=w_cam, 
#         height=h_cam,
#         paper_bgcolor='rgba(0,0,0,0)', 
#         plot_bgcolor='rgba(0,0,0,0)', # Sets background to fully transparent
#         margin=dict(l=0, r=0, t=0, b=0), # Removes all borders
#         showlegend=False, # Hiding legend so it doesn't offset the alignment
#         xaxis=dict(visible=False, range=[0, w_cam]),
#         # Crucial: Image coordinates start 0 at the top, Plotly starts 0 at the bottom.
#         # We invert the Plotly Y-axis range to perfectly match the image bounds.
#         yaxis=dict(visible=False, range=[h_cam, 0], scaleanchor="x", scaleratio=1) 
#     )
    
#     pdf_output_path = output_dir / "projected_trajectories_transparent.pdf"
#     fig_pdf.write_image(pdf_output_path)
#     print(f"Saved: {pdf_output_path}")

def project_trajectories_to_camera(system_data, output_dir: Path, camera_img_path: str):
    """Projects Vicon data onto a 2D camera image and saves both a JPG and a transparent vector PDF."""
    camera_img = cv2.imread(camera_img_path)
    if camera_img is None:
        print(f"Error: Could not load camera image from {camera_img_path}")
        return

    h_cam, w_cam = camera_img.shape[:2]

    # Vicon Coordinates relative to origin:
    pts_vicon = np.float32([
        [0.0,  2.0],   # Top Left
        [10.1, 2.0],   # Top Right
        [10.1,  -2.0],   # Bottom Right
        [0.0,  -2.0]     # Bottom Left
    ])

    # Image Pixel Coordinates:
    pts_camera = np.float32([
        [1,    835],    # Top Left Pixel Coord
        [1656, 1519],   # Top Right Pixel Coord
        [1668, 2132],   # Bottom Right Pixel Coord
        [1,    2734]    # Bottom Left Pixel Coord
    ])

    # Compute transformation matrix
    matrix = cv2.getPerspectiveTransform(pts_vicon, pts_camera)
    colors = px.colors.qualitative.Plotly
    fig_pdf = go.Figure()

    # --- 1. DRAW PERSPECTIVE GRID AND BOUNDING BOX ---
    grid_color_bgr = (200, 200, 200)
    grid_color_rgba = 'rgba(200, 200, 200, 0.5)'
    axis_color_bgr = (0, 0, 0)
    axis_color_rgba = 'rgba(0, 0, 0, 1.0)'

    # Draw Internal Grid Lines (Every 1 meter)
    for x in range(1, 10):
        pt1 = cv2.perspectiveTransform(np.float32([[[x, -2.0]]]), matrix)[0][0]
        pt2 = cv2.perspectiveTransform(np.float32([[[x,  2.0]]]), matrix)[0][0]
        cv2.line(camera_img, np.int32(pt1), np.int32(pt2), grid_color_bgr, 2, cv2.LINE_AA)
        fig_pdf.add_trace(go.Scatter(x=[pt1[0], pt2[0]], y=[pt1[1], pt2[1]], mode='lines', line=dict(color=grid_color_rgba, width=2), showlegend=False, hoverinfo='skip'))

    for y in range(-1, 2):
        pt1 = cv2.perspectiveTransform(np.float32([[[0.0,  y]]]), matrix)[0][0]
        pt2 = cv2.perspectiveTransform(np.float32([[[10.0, y]]]), matrix)[0][0]
        cv2.line(camera_img, np.int32(pt1), np.int32(pt2), grid_color_bgr, 2, cv2.LINE_AA)
        fig_pdf.add_trace(go.Scatter(x=[pt1[0], pt2[0]], y=[pt1[1], pt2[1]], mode='lines', line=dict(color=grid_color_rgba, width=2), showlegend=False, hoverinfo='skip'))

    # Draw Outer Bounding Box (Graph Edges)
    proj_corners = cv2.perspectiveTransform(pts_vicon.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    cv2.polylines(camera_img, [np.int32(proj_corners)], isClosed=True, color=axis_color_bgr, thickness=4, lineType=cv2.LINE_AA)
    fig_pdf.add_trace(go.Scatter(
        x=np.append(proj_corners[:, 0], proj_corners[0, 0]), 
        y=np.append(proj_corners[:, 1], proj_corners[0, 1]), 
        mode='lines', line=dict(color=axis_color_rgba, width=4), showlegend=False, hoverinfo='skip'
    ))

    # --- 2. ADD TICKS AND LABELS ---
    # Bottom Axis (Distance): X from 0 to 10 at Y=2
    x_tick_pts, x_tick_texts = [], []
    for x in range(0, 11):
        pt_edge = cv2.perspectiveTransform(np.float32([[[x, 2.0]]]), matrix)[0][0]
        pt_in   = cv2.perspectiveTransform(np.float32([[[x, 1.85]]]), matrix)[0][0] # Tick points inward
        
        cv2.line(camera_img, np.int32(pt_edge), np.int32(pt_in), axis_color_bgr, 4, cv2.LINE_AA)
        fig_pdf.add_trace(go.Scatter(x=[pt_edge[0], pt_in[0]], y=[pt_edge[1], pt_in[1]], mode='lines', line=dict(color=axis_color_rgba, width=4), showlegend=False, hoverinfo='skip'))
        
        x_tick_pts.append(pt_edge)
        x_tick_texts.append(f"{x}m")
        # Offset text down in purely pixel space
        cv2.putText(camera_img, f"{x}m", (int(pt_edge[0]) - 25, int(pt_edge[1]) + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, axis_color_bgr, 3, cv2.LINE_AA)

    fig_pdf.add_trace(go.Scatter(
        x=np.array(x_tick_pts)[:, 0], y=np.array(x_tick_pts)[:, 1], 
        mode='text', text=x_tick_texts, textposition='bottom center', 
        textfont=dict(color='black', size=18), showlegend=False, hoverinfo='skip'
    ))

    # Left Axis (Deviation): Y from -2 to 2 at X=0
    y_tick_pts, y_tick_texts = [], []
    for y in range(-2, 3):
        pt_edge = cv2.perspectiveTransform(np.float32([[[0.0, y]]]), matrix)[0][0]
        pt_in   = cv2.perspectiveTransform(np.float32([[[0.15, y]]]), matrix)[0][0] # Tick points inward
        
        cv2.line(camera_img, np.int32(pt_edge), np.int32(pt_in), axis_color_bgr, 4, cv2.LINE_AA)
        fig_pdf.add_trace(go.Scatter(x=[pt_edge[0], pt_in[0]], y=[pt_edge[1], pt_in[1]], mode='lines', line=dict(color=axis_color_rgba, width=4), showlegend=False, hoverinfo='skip'))
        
        y_tick_pts.append(pt_edge)
        y_tick_texts.append(f"{y}m")
        # Offset text right in purely pixel space (so it doesn't get clipped off the left side of the image)
        cv2.putText(camera_img, f"{y}m", (int(pt_edge[0]) + 30, int(pt_edge[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, axis_color_bgr, 3, cv2.LINE_AA)

    fig_pdf.add_trace(go.Scatter(
        x=np.array(y_tick_pts)[:, 0], y=np.array(y_tick_pts)[:, 1], 
        mode='text', text=y_tick_texts, textposition='middle right', 
        textfont=dict(color='black', size=18), showlegend=False, hoverinfo='skip'
    ))

    # Add Titles
    pt_mid_x = cv2.perspectiveTransform(np.float32([[[5.0, 2.0]]]), matrix)[0][0]
    cv2.putText(camera_img, "Forward Distance", (int(pt_mid_x[0]) - 160, int(pt_mid_x[1]) + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, axis_color_bgr, 4, cv2.LINE_AA)
    fig_pdf.add_trace(go.Scatter(x=[pt_mid_x[0]], y=[pt_mid_x[1] + 40], mode='text', text=["Forward Distance"], textposition='bottom center', textfont=dict(color='black', size=22, family="Arial Black"), showlegend=False, hoverinfo='skip'))

    pt_mid_y = cv2.perspectiveTransform(np.float32([[[0.0, 0.0]]]), matrix)[0][0]
    cv2.putText(camera_img, "Lateral Deviation", (int(pt_mid_y[0]) + 120, int(pt_mid_y[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, axis_color_bgr, 4, cv2.LINE_AA)
    fig_pdf.add_trace(go.Scatter(x=[pt_mid_y[0] + 40], y=[pt_mid_y[1]], mode='text', text=["Lateral Deviation"], textposition='middle right', textfont=dict(color='black', size=22, family="Arial Black"), showlegend=False, hoverinfo='skip'))

    # --- 3. DRAW TRAJECTORIES ---
    for idx, (raw_sys_name, data_arrays) in enumerate(system_data.items()):
        hex_color = colors[idx % len(colors)]
        bgr_color = hex_to_bgr(hex_color)
        fill_color = hex_to_rgba(hex_color, 0.2)
        clean_name = POLICY_NAME_MAP.get(raw_sys_name, raw_sys_name)

        mean_x = np.mean(data_arrays['x'], axis=0)
        mean_y = np.mean(data_arrays['y'], axis=0)
        std_y = np.std(data_arrays['y'], axis=0)
        
        upper_bound = mean_y + std_y
        lower_bound = mean_y - std_y

        # Variance Region
        upper_pts = np.vstack((mean_x, upper_bound)).T
        lower_pts = np.vstack((mean_x, lower_bound)).T
        polygon_pts = np.vstack((upper_pts, lower_pts[::-1])).reshape(-1, 1, 2).astype(np.float32)
        projected_polygon = cv2.perspectiveTransform(polygon_pts, matrix)
        
        overlay = camera_img.copy()
        cv2.fillPoly(overlay, [np.int32(projected_polygon)], color=bgr_color)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, camera_img, 1 - alpha, 0, camera_img)

        fig_pdf.add_trace(go.Scatter(
            x=projected_polygon[:, 0, 0], y=projected_polygon[:, 0, 1], 
            mode='lines', line=dict(width=0), fill='toself', 
            fillcolor=fill_color, showlegend=False, hoverinfo='skip'
        ))

        # Mean Line
        vicon_pts = np.vstack((mean_x, mean_y)).T.reshape(-1, 1, 2).astype(np.float32)
        projected_mean = cv2.perspectiveTransform(vicon_pts, matrix)
        
        cv2.polylines(camera_img, [np.int32(projected_mean)], isClosed=False, color=bgr_color, thickness=4, lineType=cv2.LINE_AA)

        fig_pdf.add_trace(go.Scatter(
            x=projected_mean[:, 0, 0], y=projected_mean[:, 0, 1], 
            mode='lines', name=clean_name, line=dict(color=hex_color, width=3)
        ))

    # --- 4. SAVE OUTPUTS ---
    img_output_path = output_dir / "projected_camera_view.jpg"
    cv2.imwrite(str(img_output_path), camera_img)
    print(f"Saved: {img_output_path}")

    fig_pdf.update_layout(
        width=w_cam, height=h_cam, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        xaxis=dict(visible=False, range=[0, w_cam]),
        yaxis=dict(visible=False, range=[h_cam, 0], scaleanchor="x", scaleratio=1) 
    )
    
    pdf_output_path = output_dir / "projected_trajectories_transparent.pdf"
    fig_pdf.write_image(pdf_output_path)
    print(f"Saved: {pdf_output_path}")


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

    if FLAGS.camera_image:
        print("Projecting onto camera view...")
        project_trajectories_to_camera(system_data, base_dir, FLAGS.camera_image)

    print("\nPlotting complete.")


if __name__ == '__main__':
    app.run(main)
