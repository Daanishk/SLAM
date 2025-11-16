import numpy as np
import open3d as o3d
from PIL import Image
import cv2
import glob
import os

def cleanup_snapshots():
    patterns = ["color_*.png", "depth_*.png", "pose_*.npy", "position_*.npy"]
    for pattern in patterns:
        for fn in glob.glob(pattern):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

def build_merged_cloud_from_snapshots():
    pos_files = sorted(glob.glob("position_*.npy"))
    if not pos_files:
        raise RuntimeError("No position_*.npy files found.")

    all_points = []
    all_colors = []

    for pos_path in pos_files:
        stem = os.path.splitext(os.path.basename(pos_path))[0]   # e.g., 'depth_000'
        suffix = stem.split("_")[1]                                # '000'

        color_path = f"color_{suffix}.png"
        pose_path = f"pose_{suffix}.npy"

        position = np.load(pos_path)       
        valid = position[..., 3] < 1
        pts_opengl = position[..., :3][valid]   

        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0
        cols_valid = color[valid]

        model_matrix = np.load(pose_path) 
        R = model_matrix[:3, :3]
        t = model_matrix[:3, 3]
        pts_world = pts_opengl @ R.T + t

        all_points.append(pts_world)
        all_colors.append(cols_valid)

    pts_merged = np.concatenate(all_points, axis=0)
    cols_merged = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_merged.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols_merged.astype(np.float64))

    return pcd


def visualize_merged_snapshots(cleanup = True):
    pcd = build_merged_cloud_from_snapshots()
    o3d.visualization.draw_geometries([pcd])
    if cleanup:
        cleanup_snapshots()