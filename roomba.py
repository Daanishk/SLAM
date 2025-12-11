import sapien
import trimesh
from sapien.utils.viewer import Viewer
import numpy as np
import cv2
from PIL import Image, ImageColor

from floor import Floor

class Roomba:
    def __init__(self, scene, viewer):
        self.scene = scene
        self.viewer = viewer

        ## Camera Constants
        near, far = 0.1, 100
        width, height = 640, 480
        half_size = np.array([0.1, 0.1, 0.1])

        # idx for file saving
        self.snapshot_idx = 0

        # Create Roomba with attached actor

        actor_builder = scene.create_actor_builder()
        # @TODO: Make it a cylinder. Cylinders are by default in the direction of x rather than z, so not trivial
        actor_builder.add_box_collision(half_size=half_size)
        actor_builder.add_box_visual(half_size=half_size, material=np.array([1,0,0]))
        self.entity = actor_builder.build(name="roomba")
        self.camera = scene.add_mounted_camera(
            name="mounted_camera",
            mount=self.entity,
            pose=sapien.Pose(np.eye(4)),
            width=width,
            height=height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        self.camera_body = self.entity.get_components()[1]

    def get_pose_as_transformation_matrix(self):
        return np.array(self.camera.get_entity_pose().to_transformation_matrix())

    def move_forward(self):
        self.move(np.array([1,0,0]))

    def move(self, direction):
        self.camera_body.set_linear_velocity(self.get_pose_as_transformation_matrix()[:3,:3] @ direction)

    def turn_left(self):
        self.camera_body.set_angular_velocity(np.array([0,0,1]))
    
    def turn_right(self):
        self.camera_body.set_angular_velocity(np.array([0,0,-1]))

    def stop(self):
        self.camera_body.set_linear_velocity(np.array([0,0,0]))

    def take_snapshot(self):
        self.camera.take_picture()
        idx_str = f"{self.snapshot_idx:03d}" 

        rgba = self.camera.get_picture("Color")  # [H, W, 4]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)
        rgba_pil.save(f"snapshots/color_{idx_str}.png")

        position = self.camera.get_picture("Position")  # [H, W, 4]
        np.save(f"snapshots/position_{idx_str}.npy", position)

        model_matrix = self.camera.get_model_matrix()
        np.save(f"snapshots/pose_{idx_str}.npy", model_matrix)

        print(f"Saved snapshot {idx_str}")
        self.snapshot_idx += 1

        return rgba_img

    def load_snapshot(self, idx):
        pos_path = f"snapshots/position_{idx}.npy"
        position = np.load(pos_path)
        pose_path = f"snapshots/pose_{idx}.npy"
        pose = np.load(pose_path)

        color_path = f"snapshots/color_{idx}.png"
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rgba = color.astype(np.float32) / 255.0

        # OpenGL/Blender: y up and -z forward
        points_opengl = position[..., :3][position[..., 3] < 1]
        points_color = rgba[position[..., 3] < 1]
        # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
        points_world = points_opengl @ pose[:3, :3].T + pose[:3, 3]
        return points_world, points_color

    def visualize_depth_picture(self, points_world, points_color):
        points_color = (np.clip(points_color, 0, 1) * 255).astype(np.uint8)
        trimesh.PointCloud(points_world, points_color).show()
        
    def set_controller(self, controller):
        self.controller = controller

    def perform_action(self, timestep):
        if self.controller is None:
            return
        else:
            self.controller.next(self, timestep)

class RoombaController():
    def next(self, roomba, timestep):
        pass

class ManualController(RoombaController):
    def next(self, roomba, timestep):
        if roomba.viewer.window.key_press("p"):  # Press 'p' to take the screenshot
            roomba.take_snapshot()
            self.handle_snapshot(roomba)
        camera_pose = roomba.camera.get_entity_pose()
        camera_mat = np.array(camera_pose.to_transformation_matrix())
        direction = np.array([0,0,0])
        if roomba.viewer.window.key_down("i"):
            direction += np.array([1,0,0])
        if roomba.viewer.window.key_down("j"):
            direction += np.array([0,1,0])
        if roomba.viewer.window.key_down("k"):
            direction += np.array([-1,0,0])
        if roomba.viewer.window.key_down("l"):
            direction += np.array([0,-1,0])
        if roomba.viewer.window.key_down("u"):
            roomba.turn_left()
        if roomba.viewer.window.key_down("o"):
            roomba.turn_right()
        if roomba.viewer.window.key_down("m"):
            roomba.viewer.set_camera_pose(sapien.Pose(camera_mat))
        roomba.move(direction)

    def handle_snapshot(self, roomba):
        pass


class TrajectoryController(RoombaController):
    def __init__(self, points):
        self.points = points
        self.objective_index = 0
        self.completed = False
    
    def next_objective(self):
        self.objective_index += 1
        if (self.objective_index >= len(self.points)):
            self.completed = True

    def next(self, roomba, timestep):
        if self.completed:
            return
        
        curr_point = self.points[self.objective_index][:2]
        mat = roomba.get_pose_as_transformation_matrix()
        roomba_dir = mat[:2,:2] @ np.array([1,0])
        roomba_pos = mat[:2,3]

        roomba_to_point = curr_point - roomba_pos
        if (np.linalg.norm(roomba_to_point) < .1):
            self.next_objective()
            return

        roomba_to_point = roomba_to_point / np.linalg.norm(roomba_to_point)
        cos_theta = np.dot(roomba_dir / np.linalg.norm(roomba_dir), roomba_to_point)
        theta = np.arccos(cos_theta)
        if theta > np.pi / 2:
            theta = theta - np.pi
        theta_degrees = theta * 180 / np.pi
        # @TODO: Make this work (i.e. turn left as well) and/or do a learning-based approach
        if np.abs(theta_degrees) > 5 or cos_theta < 0:
            roomba.turn_right()
        else:
            roomba.move_forward()

class ManualSlamController(ManualController):
    def __init__(self):
        self.point_cloud = None
        self.point_cloud_colors = None
        self.ground_threshold = 0.1
        self.floor = Floor(1,1,5)
        self.floor.init_tiles(30, 30)

    def handle_snapshot(self, roomba):
        idx = f"{(roomba.snapshot_idx-1):03d}" 
        pts, color = roomba.load_snapshot(idx)
        points_above_ground = np.where(pts[:, 2] > self.ground_threshold)
        pts = pts[points_above_ground]
        color = color[points_above_ground]

        # Iterate through points in point cloud
        for i in range(pts.shape[0]):
            point = pts[i]
            pixel = self.convert_point_to_pixel(point)

            # @TODO: Things aren't quite correct here
            # # If pixel is left of it, then we need to move it to the right by 1
            # if pixel[1] - roomba_pos[1] <= 0:
            #     pixel -= np.array([0,1])

            # # If pixel is above it, then we need to move it up by 1
            # if pixel[0] - roomba_pos[0] < 0:
            #     pixel -= np.array([1,0])

            pixel = np.array([15, 15]) + pixel

            # @TODO Has trouble with doorways

            pixel_color = (color[i,:3] * 255).astype(np.uint8)
            
            # Has orange color, I think from the camera visual
            if pixel_color[0] in [235, 236] and pixel_color[1] in [146,147,148] and pixel_color[2] in [47, 48]:
                continue
            self.floor.tiles[pixel[0],pixel[1]] = pixel_color
        
        self.floor.save_image("slam_floor.png")

    def convert_point_to_pixel(self, point):
        pixel = np.rint(point / np.array([self.floor.tile_width, self.floor.tile_length, 1]))
        return np.int32(pixel[:2])
