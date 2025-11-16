import sapien
import trimesh
from sapien.utils.viewer import Viewer
import numpy as np
from PIL import Image, ImageColor

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
        rgba_pil.save(f"color_{idx_str}.png")

        position = self.camera.get_picture("Position")  # [H, W, 4]
        np.save(f"position_{idx_str}.npy", position)

        model_matrix = self.camera.get_model_matrix()
        np.save(f"pose_{idx_str}.npy", model_matrix)

        print(f"Saved snapshot {idx_str}")
        self.snapshot_idx += 1


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

class SlamController(RoombaController):
    def next(self, roomba, timestep):
        # @TODO: Do this.
        pass
    