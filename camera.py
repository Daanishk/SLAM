"""Camera.

Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
"""

import sapien
import numpy as np
from PIL import Image, ImageColor

import trimesh
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler
import json

def main():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 1], [1, 1, 1])
    
    with open('sample.json') as f:
        floor = json.load(f)
        for key in floor.keys():
            room = floor[key]
            width = room["width"]
            height = 5
            position = np.array([room["pos_x"],room["pos_y"], 0])
            color = np.array([255, 0, 0])

            wall1_half_size = np.array([room["size_x"] - width, width, height])/2
            wall1_center = position + np.array([width, width, 0]) + wall1_half_size
            wall1_pose = sapien.Pose(wall1_center)
            wall1_builder: sapien.ActorBuilder = scene.create_actor_builder()
            wall1_builder.add_box_collision(half_size=wall1_half_size)  # Add collision shape
            wall1_builder.add_box_visual(half_size=wall1_half_size, material=color)  # Add visual shape
            wall1_box: sapien.Entity = wall1_builder.build(name=key + "wall1")
            wall1_box.set_pose(wall1_pose)

            wall2_half_size = np.array([room["size_x"] - width, width, height])/2
            wall2_center = position + wall2_half_size + np.array([0, room["size_y"] - width, 0])
            wall2_pose = sapien.Pose(wall2_center)
            wall2_builder: sapien.ActorBuilder = scene.create_actor_builder()
            wall2_builder.add_box_collision(half_size=wall2_half_size)  # Add collision shape
            wall2_builder.add_box_visual(half_size=wall2_half_size, material=color)  # Add visual shape
            wall2_box: sapien.Entity = wall2_builder.build(name=key + "wall2")
            wall2_box.set_pose(wall2_pose)

            wall3_half_size = np.array([width, room["size_y"] - width, height])/2
            wall3_center = position + np.array([width, width, 0]) + wall3_half_size
            wall3_pose = sapien.Pose(wall3_center)
            wall3_builder: sapien.ActorBuilder = scene.create_actor_builder()
            wall3_builder.add_box_collision(half_size=wall3_half_size)  # Add collision shape
            wall3_builder.add_box_visual(half_size=wall3_half_size, material=color)  # Add visual shape
            wall3_box: sapien.Entity = wall3_builder.build(name=key + "wall3")
            wall3_box.set_pose(wall3_pose)

            wall4_half_size = np.array([width, room["size_y"] - width, height])/2
            wall4_center = position + wall4_half_size + np.array([room["size_x"], 0, 0])
            wall4_pose = sapien.Pose(wall4_center)
            wall4_builder: sapien.ActorBuilder = scene.create_actor_builder()
            wall4_builder.add_box_collision(half_size=wall4_half_size)  # Add collision shape
            wall4_builder.add_box_visual(half_size=wall4_half_size, material=color)  # Add visual shape
            wall4_box: sapien.Entity = wall4_builder.build(name=key + "wall4")
            wall4_box.set_pose(wall4_pose)

    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    width, height = 640, 480

    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    cam_pos = np.array([-2, -2, 3])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )
    camera.entity.set_pose(sapien.Pose([0,1,0]))

    print("Intrinsic matrix\n", camera.get_intrinsic_matrix())

    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    mounted_camera = scene.add_mounted_camera(
        name="mounted_camera",
        mount=camera_mount_actor,
        pose=sapien.Pose(mat44),
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )

    # scene.step()  # run a physical step
    # scene.update_render()  # sync pose from SAPIEN to renderer
    # camera.take_picture()  # submit rendering jobs to the GPU

    # # ---------------------------------------------------------------------------- #
    # # RGBA
    # # ---------------------------------------------------------------------------- #
    # rgba = camera.get_picture("Color")  # [H, W, 4]
    # rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    # rgba_pil = Image.fromarray(rgba_img)
    # rgba_pil.save("color.png")

    # # ---------------------------------------------------------------------------- #
    # # XYZ position in the camera space
    # # ---------------------------------------------------------------------------- #
    # # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
    # position = camera.get_picture("Position")  # [H, W, 4]

    # # OpenGL/Blender: y up and -z forward
    # points_opengl = position[..., :3][position[..., 3] < 1]
    # points_color = rgba[position[..., 3] < 1]
    # # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
    # # camera.get_model_matrix() must be called after scene.update_render()!
    # model_matrix = camera.get_model_matrix()
    # points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]

    # # SAPIEN CAMERA: z up and x forward
    # # points_camera = points_opengl[..., [2, 0, 1]] * [-1, -1, 1]

    # points_color = (np.clip(points_color, 0, 1) * 255).astype(np.uint8)
    # trimesh.PointCloud(points_world, points_color).show()

    # # Depth
    # depth = -position[..., 2]
    # depth_image = (depth * 1000.0).astype(np.uint16)
    # depth_pil = Image.fromarray(depth_image)
    # depth_pil.save("depth.png")

    # # ---------------------------------------------------------------------------- #
    # # Segmentation labels
    # # ---------------------------------------------------------------------------- #
    # # Each pixel is (visual_id, actor_id/link_id, 0, 0)
    # # visual_id is the unique id of each visual shape
    # seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
    # colormap = sorted(set(ImageColor.colormap.values()))
    # color_palette = np.array(
    #     [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
    # )
    # label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    # label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    # # Or you can use aliases below
    # # label0_image = camera.get_visual_segmentation()
    # # label1_image = camera.get_actor_segmentation()
    # label0_pil = Image.fromarray(color_palette[label0_image])
    # label0_pil.save("label0.png")
    # label1_pil = Image.fromarray(color_palette[label1_image])
    # label1_pil.save("label1.png")

    # ---------------------------------------------------------------------------- #
    # Take picture from the viewer
    # ---------------------------------------------------------------------------- #
    viewer = Viewer()
    viewer.set_scene(scene)
    # We show how to set the viewer according to the pose of a camera
    # opengl camera -> sapien world
    model_matrix = camera.get_model_matrix()
    # sapien camera -> sapien world
    # You can also infer it from the camera pose
    model_matrix = model_matrix[:, [2, 0, 1, 3]] * np.array([-1, -1, 1, 1])
    # The rotation of the viewer camera is represented as [roll(x), pitch(-y), yaw(-z)]
    rpy = mat2euler(model_matrix[:3, :3]) * np.array([1, -1, -1])
    viewer.set_camera_xyz(*model_matrix[0:3, 3])
    viewer.set_camera_rpy(*rpy)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    while not viewer.closed:
        scene.step()
        scene.update_render()
        if viewer.window.key_down("p"):  # Press 'p' to take the screenshot
            camera.take_picture()  # submit rendering jobs to the GPU
            rgba = camera.get_picture("Color")  # [H, W, 4]
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            rgba_pil.save("color.png")
        camera_pose = camera.entity.get_pose()
        camera_pos = np.array(camera_pose.get_p())
        camera_rot = np.array(camera_pose.get_rpy())
        camera_mat = np.eye(4)
        if viewer.window.key_down("i"):
            camera.entity.set_pose(sapien.Pose(camera_pos + np.array([1,0,0])))
        if viewer.window.key_down("j"):
            camera.entity.set_pose(sapien.Pose(camera_pos + np.array([0,-1,0])))
        if viewer.window.key_down("k"):
            camera.entity.set_pose(sapien.Pose(camera_pos + np.array([-1,0,0])))
        if viewer.window.key_down("l"):
            camera.entity.set_pose(sapien.Pose(camera_pos + np.array([0,1,0])))
        viewer.render()


if __name__ == "__main__":
    main()
