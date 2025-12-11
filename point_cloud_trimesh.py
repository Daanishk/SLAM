import numpy as np
import trimesh
from PIL import Image
import cv2
import glob
import os

def cleanup_snapshots():
    patterns = ["color_*.png", "depth_*.png", "pose_*.npy", "position_*.npy"]
    for pattern in patterns:
        for fn in glob.glob("snapshots/" + pattern):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

def build_merged_cloud_from_snapshots():
    pos_files = sorted(glob.glob("snapshots/position_*.npy"))
    if not pos_files:
        raise RuntimeError("No position_*.npy files found.")

    all_points = []
    all_colors = []

    for pos_path in pos_files:
        stem = os.path.splitext(os.path.basename(pos_path))[0]   # e.g., 'depth_000'
        suffix = stem.split("_")[1]                                # '000'

        color_path = f"snapshots/color_{suffix}.png"
        pose_path = f"snapshots/pose_{suffix}.npy"

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

    cols_merged = (np.clip(cols_merged, 0, 1) * 255).astype(np.uint8)

    return pts_merged, cols_merged


def visualize_merged_snapshots(cleanup = True):
    pts_merged, cols_merged = build_merged_cloud_from_snapshots()
    trimesh.PointCloud(pts_merged, cols_merged).show()
    if cleanup:
        cleanup_snapshots()