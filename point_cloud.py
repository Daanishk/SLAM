import numpy as np
import open3d as o3d
from PIL import Image
import cv2

def get_camera_intrinsics_from_sapien(width, height, fovy_deg=35.0):
    fovy = np.deg2rad(fovy_deg)
    fx = fy = 0.5 * width / np.tan(fovy / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1 ]])
    return K


def depth_to_camera_frame_point_cloud(depth, K):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    mask = (depth > 0) & np.isfinite(depth)

    x_valid = x[mask]
    y_valid = y[mask]
    z_valid = depth[mask]

    flat = np.stack([x_valid, y_valid, np.ones_like(x_valid)], axis=-1).T
    K_inv = np.linalg.inv(K)
    rays = K_inv @ flat
    pts_cam = (rays * z_valid).T    # N x 3

    return pts_cam, mask


def make_o3d_cloud_from_files(depth_path="depth.png", color_path="color.png"):
    # depth: uint16 in mm
    depth_mm = np.array(Image.open(depth_path)).astype(np.float32)
    depth = depth_mm / 1000.0  # m
    H, W = depth.shape

    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color = color.astype(np.float32) / 255.0

    K = get_camera_intrinsics_from_sapien(W, H, fovy_deg=35.0)

    pts_cam, mask = depth_to_camera_frame_point_cloud(depth, K)
    colors_valid = color[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cam.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_valid.astype(np.float64))
    return pcd

def visualize_roomba_cloud(depth_path="depth.png", color_path="color.png"):
    pcd = make_o3d_cloud_from_files(depth_path, color_path)
    o3d.visualization.draw_geometries([pcd])