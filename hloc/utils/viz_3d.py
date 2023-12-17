"""
3D visualization based on plotly.
Works for a small number of points and cameras, might be slow otherwise.

1) Initialize a figure with `init_figure`
2) Add 3D points, camera frustums, or both as a pycolmap.Reconstruction

Written by Paul-Edouard Sarlin and Philipp Lindenberger.
"""

from typing import Optional
import numpy as np
import pycolmap
import plotly.graph_objects as go
import open3d as o3d
import os
from pathlib import PurePosixPath
from tqdm import tqdm

def to_homogeneous(points):
    pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="simple_white",
        height=height,
        scene_camera=dict(
            eye=dict(x=0., y=-.1, z=-2),
            up=dict(x=0, y=-1., z=0),
            projection=dict(type="orthographic")),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode='data',
            dragmode='orbit',
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.1
        ),
    )
    return fig


def plot_points(
        fig: go.Figure,
        pts: np.ndarray,
        color: str = 'rgba(255, 0, 0, 1)',
        ps: int = 2,
        colorscale: Optional[str] = None,
        name: Optional[str] = None):
    """Plot a set of 3D points."""
    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x, y=y, z=z, mode='markers', name=name, legendgroup=name,
        marker=dict(
            size=ps, color=color, line_width=0.0, colorscale=colorscale))
    fig.add_trace(tr)


def plot_camera(
        fig: go.Figure,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        color: str = 'rgb(0, 0, 255)',
        name: Optional[str] = None,
        legendgroup: Optional[str] = None,
        fill: bool = False,
        size: float = 1.0,
        text: Optional[str] = None):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2]*2, K[1, 2]*2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t
    legendgroup = legendgroup if legendgroup is not None else name

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    if fill:
        pyramid = go.Mesh3d(
            x=x, y=y, z=z, color=color, i=i, j=j, k=k,
            legendgroup=legendgroup, name=name, showlegend=False,
            hovertemplate=text.replace('\n', '<br>'))
        fig.add_trace(pyramid)

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([
        vertices[i] for i in triangles.reshape(-1)
    ])
    x, y, z = tri_points.T

    pyramid = go.Scatter3d(
        x=x, y=y, z=z, mode='lines', legendgroup=legendgroup,
        name=name, line=dict(color=color, width=1), showlegend=False,
        hovertemplate=text.replace('\n', '<br>'))
    fig.add_trace(pyramid)


def plot_camera_colmap(
        fig: go.Figure,
        image: pycolmap.Image,
        camera: pycolmap.Camera,
        name: Optional[str] = None,
        **kwargs):
    """Plot a camera frustum from PyCOLMAP objects"""
    plot_camera(
        fig,
        image.rotmat().T,
        image.projection_center(),
        camera.calibration_matrix(),
        name=name or str(image.image_id),
        text=image.summary(),
        **kwargs)


def plot_cameras(
        fig: go.Figure,
        reconstruction: pycolmap.Reconstruction,
        **kwargs):
    """Plot a camera as a cone with camera frustum."""
    for image_id, image in reconstruction.images.items():
        plot_camera_colmap(
            fig, image, reconstruction.cameras[image.camera_id], **kwargs)


def plot_reconstruction(
        fig: go.Figure,
        rec: pycolmap.Reconstruction,
        max_reproj_error: float = 6.0,
        color: str = 'rgb(0, 0, 255)',
        p_color: str = 'rgb(95, 70, 50)',
        name: Optional[str] = None,
        min_track_length: int = 2,
        points: bool = True,
        cameras: bool = True,
        points_rgb: bool = True,
        cs: float = 1.0,
        pcd_save_path: str = "."):
    # Filter outliers
    bbs = rec.compute_bounding_box(0.001, 0.999)
    # Filter points, use original reproj error here
    p3Ds = [p3D for _, p3D in rec.points3D.items() if (
                            (p3D.xyz >= bbs[0]).all() and
                            (p3D.xyz <= bbs[1]).all() and
                            p3D.error <= max_reproj_error and
                            p3D.track.length() >= min_track_length)]
    xyzs = np.array([p3D.xyz for p3D in p3Ds])
    print("original point cloud shape:", xyzs.shape)
    #xyzs = postprocess_pcd(xyzs, pcd_save_path, nb_points=3, radius=0.05)
    #print("postprocessed point cloud shape:", xyzs.shape)
    save_pcd(xyzs, pcd_save_path)
    if points_rgb:
        pcolor = [p3D.color for p3D in p3Ds]
    else:
        pcolor = p_color
    if points:
        plot_points(fig, xyzs, color=pcolor, ps=1, name=name)
    if cameras:
        plot_cameras(fig, rec, color=color, legendgroup=name, size=cs)

def save_cameras(rec: pycolmap.Reconstruction, save_camera_path: str = "."):
    print(f"Saving cameras to {save_camera_path}")
    os.makedirs(save_camera_path, exist_ok=True)
    for image_id, image in tqdm(rec.images.items()):
        camera = rec.cameras[image.camera_id]
        R = image.rotmat().T
        t = image.projection_center()
        K = camera.calibration_matrix()
        RT = np.zeros((3,4))
        RT[:,:3] = R.reshape((3,3))
        RT[:,3] = t.flatten()
        camera_name = PurePosixPath(str(image.name)).stem
        res_out = {
            "K": np.array(K).astype(np.float64),
            "RT": np.array(RT).astype(np.float64),
        }
        cam_matrix_save_path = os.path.join(save_camera_path, f"{camera_name}.npz")
        np.savez(cam_matrix_save_path, **res_out)
        

def postprocess_pcd(pts_in, pcd_save_path, nb_points=7, radius=0.05):
    """
    [Not in use] Array of shape (n, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_in)
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_cloud = cl.select_by_index(ind)
    o3d.io.write_point_cloud(f"{pcd_save_path}/reconstructed.ply", inlier_cloud)
    pts_out = np.asarray(inlier_cloud.points)
    return pts_out

def save_pcd(pts_in, pcd_save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_in)
    o3d.io.write_point_cloud(f"{pcd_save_path}/reconstructed.pcd", pcd)