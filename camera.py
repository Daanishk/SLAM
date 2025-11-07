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
import cv2

def main():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 1], [1, 1, 1])
    image = cv2.imread("floor.png", cv2.IMREAD_COLOR)
    unique_colors = np.unique(image.reshape(-1, image.shape[-1]), axis=0)

    width = 1
    height = 5

    img_shape = np.array(image.shape)
    img_shape[2] = 0
    pixel_map = image[:,:,0] * (256.0 ** 2) + image[:,:,1] * 256.0 + image[:,:,2]
    for color in unique_colors:
        # Ignore Background
        if np.sum(color) == 255 + 255 + 255:
            continue
        color_rep = color[0] * (256.0 ** 2) + color[1] * 256.0 + color[2]
        pixels = np.argwhere(pixel_map == color_rep)
        pixels = pixels[:,:2]
        pixels = np.unique(pixels, axis=0)
        for pixel in pixels:
            position = np.array([pixel[0], pixel[1], 0]) * width
            # Theoretically, could add all walls as part of same Entity, but not sure if actually needed
            wall_half_size = np.array([width, width, height])/2
            wall_center = position + np.array([width, width, height])/2 - (img_shape / 2)
            wall_pose = sapien.Pose(wall_center)
            wall_builder: sapien.ActorBuilder = scene.create_actor_builder()
            wall_builder.add_box_collision(half_size=wall_half_size)  # Add collision shape
            wall_builder.add_box_visual(half_size=wall_half_size, material=color / 255)  # Add visual shape
            wall_builder.set_initial_pose(wall_pose)
            wall_box: sapien.Entity = wall_builder.build(name=str(pixel))
            shapes = wall_box.get_components()

            # Set collision group so they all don't collide with each other
            shapes[1].get_collision_shapes()[0].set_collision_groups([1,1,1,1])

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

    # print("Intrinsic matrix\n", camera.get_intrinsic_matrix())

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
        camera_mat = np.array(camera_pose.to_transformation_matrix())
        new_mat = np.eye(4)
        if viewer.window.key_down("i"):
            new_mat[:3, 3] += np.array([1,0,0]) / 10
        if viewer.window.key_down("j"):
            new_mat[:3, 3] += np.array([0,1,0]) / 10
        if viewer.window.key_down("k"):
            new_mat[:3, 3] += np.array([-1,0,0]) / 10
        if viewer.window.key_down("l"):
            new_mat[:3, 3] += np.array([0,-1,0]) / 10
        if viewer.window.key_down("u"):
            new_mat[:2, :2] += np.array([[0,-1],[1,0]]) / 10
        if viewer.window.key_down("o"):
            new_mat[:2, :2] += np.array([[0,1],[-1,0]]) / 10
        camera.entity.set_pose(sapien.Pose(np.matmul(camera_mat, new_mat)))
        if viewer.window.key_down("m"):
            viewer.set_camera_pose(sapien.Pose(np.matmul(camera_mat, new_mat)))
        viewer.render()


if __name__ == "__main__":
    main()
