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
from roomba import *
from floor import *
from point_cloud_trimesh import *

def create_scene():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_point_light([-1, 0, 1], [1, 1, 1])
    return scene

def load_floor_plan(floor, scene):
    image = floor.tiles
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

def sync_viewer_to_roomba(viewer, roomba):
    # # OpenGL cam -> SAPIEN world
    # model_matrix = roomba.camera.get_model_matrix()
    # # SAPIEN cam -> SAPIEN world
    # model_matrix = model_matrix[:, [2, 0, 1, 3]] * np.array([-1, -1, 1, 1])

    # # Viewer uses [roll(x), pitch(-y), yaw(-z)]
    # rpy = mat2euler(model_matrix[:3, :3]) * np.array([1, -1, -1])

    # viewer.set_camera_xyz(*model_matrix[0:3, 3])
    # viewer.set_camera_rpy(*rpy)

    # We can just set the pose of the viewer camera to the pose of the camera without having to multiply    
    camera_pose = roomba.camera.get_entity_pose()
    camera_mat = np.array(camera_pose.to_transformation_matrix())
    viewer.set_camera_pose(sapien.Pose(camera_mat))

def main():
    scene = create_scene()

    floor = Floor(1, 1, 5)
    floor.set_image("floor.png")
    load_floor_plan(floor, scene)

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
    # label1_image = seg
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

    half_size = np.array([0.1, 0.1, 0.1])

    roomba = Roomba(scene, viewer)

    controller = FrontierExplorationController()
    roomba.set_controller(controller)

    # We show how to set the viewer according to the pose of a camera
    # sync_viewer_to_roomba(viewer, roomba)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    while not viewer.closed:
        scene.step()
        scene.update_render()
        roomba.perform_action(scene.get_timestep())

        #syncing viewer to roomba 
        # sync_viewer_to_roomba(viewer, roomba)ppp

        viewer.render()

    visualize_merged_snapshots()

if __name__ == "__main__":
    main()
