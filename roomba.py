import sapien
import trimesh
from sapien.utils.viewer import Viewer
import numpy as np
import cv2
from PIL import Image, ImageColor
import heapq

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
            fovy=np.deg2rad(60),
            near=near,
            far=far,
        )
        self.camera_body = self.entity.get_components()[1]

    def get_pose_as_transformation_matrix(self):
        return np.array(self.camera.get_entity_pose().to_transformation_matrix())

    def move_forward(self):
        self.camera_body.set_angular_velocity(np.array([0,0,0]))
        self.move(np.array([1,0,0]))

    def move(self, direction):
        self.camera_body.set_linear_velocity(self.get_pose_as_transformation_matrix()[:3,:3] @ direction)

    def turn_left(self):
        self.camera_body.set_angular_velocity(np.array([0,0,1]))
    
    def turn_right(self):
        self.camera_body.set_angular_velocity(np.array([0,0,-1]))

    def stop(self):
        self.camera_body.set_linear_velocity(np.array([0,0,0]))
        self.camera_body.set_angular_velocity(np.array([0,0,0]))

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

        print(f"Saved snapshot {idx_str} \n")
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

        # Checking to see if we've reached the waypoint
        if (np.linalg.norm(roomba_to_point) < .1):
            self.next_objective()
            return

        roomba_to_point = roomba_to_point / np.linalg.norm(roomba_to_point)

        # Obtaining rotation angle [-2pi, 2pi]
        target_theta = np.arctan2(roomba_to_point[1], roomba_to_point[0])
        roomba_theta = np.arctan2(roomba_dir[1], roomba_dir[0])
        theta = target_theta - roomba_theta

        # making sure to obtain smallest absolute angle
        if (abs(theta) > np.pi):
            theta = theta - (np.sign(theta) * 2 * np.pi)
        theta_degrees = theta * 180 / np.pi

        # Updating direction depending on magnitude of angle
        epsilon = 5
        if theta_degrees > epsilon:
            roomba.turn_left()
        elif theta_degrees < -epsilon:
            roomba.turn_right()
        else:
            roomba.move_forward()

class ManualSlamController(ManualController):
    UNKNOWN_COLOR = np.array([127, 127, 127], dtype=np.uint8)
    FREE_COLOR = np.array([255, 255, 255], dtype=np.uint8)
    OCC_COLOR = np.array([0,   0,   0  ], dtype=np.uint8)

    def __init__(self):
        self.point_cloud = None
        self.point_cloud_colors = None
        self.ground_threshold = 0.1
        self.floor = Floor(1,1,5)
        self.width = 30
        self.height = 30
        self.center = np.array([self.height // 2, self.width // 2])
        self.floor.init_tiles(self.width, self.height)
        self.frontier_cells = None

    def in_bounds(self, rc):
        r, c = rc
        return 0 <= r < self.height and 0 <= c < self.width
    
    def get_roomba_pixel_position(self, roomba):
        mat = roomba.get_pose_as_transformation_matrix()
        roomba_pos_world = mat[:2, 3]        
        roomba_pos_3d = np.array([roomba_pos_world[0],
                                  roomba_pos_world[1],
                                  0.0], dtype=np.float32)

        # roomba cell in grid
        roomba_pixel = self.convert_point_to_pixel(roomba_pos_3d)
        roomba_pixel = self.center + roomba_pixel

        return roomba_pixel
        
    # handle_snapshot does numerous things:
    # - collects relevantly indexed snapshot; filters out wall pixels; 
    # get's the roomba's position and sets it as free in the occupancy map;
    # Goes through all the valid pixels and populates the tile and occupancy array
    def handle_snapshot(self, roomba):
        idx = f"{(roomba.snapshot_idx-1):03d}" 
        pts, color = roomba.load_snapshot(idx)
        points_above_ground = np.where(pts[:, 2] > self.ground_threshold)
        pts = pts[points_above_ground]
        color = color[points_above_ground]

        # Getting roomba postion
        roomba_pixel = self.get_roomba_pixel_position(roomba)

        # mark roomba cell itself as free
        if self.in_bounds(roomba_pixel):
            self.floor.occupancy[roomba_pixel[0], roomba_pixel[1]] = self.FREE_COLOR

        # Iterate through points in point cloud
        for i in range(pts.shape[0]):
            point = pts[i]
            pixel = self.convert_point_to_pixel(point)
            pixel = self.center + pixel

            if not self.in_bounds(pixel):
                continue

            pixel_color = (color[i,:3] * 255).astype(np.uint8)
            
            # Has orange color, I think from the camera visual
            if pixel_color[0] in [235, 236] and pixel_color[1] in [146,147,148] and pixel_color[2] in [47, 48]:
                continue

            # line of cells from robot to this wall cell
            line_cells = self.bresenham_line(roomba_pixel, pixel)

            # all but last cell = free if not already occupied
            for r, c in line_cells[:-1]:
                if not self.in_bounds((r, c)):
                    continue
                if np.array_equal(self.floor.occupancy[r, c], self.UNKNOWN_COLOR):
                    self.floor.occupancy[r, c] = self.FREE_COLOR

            # last cell = occupied (wall)
            end_r, end_c = line_cells[-1]
            if self.in_bounds((end_r, end_c)):
                self.floor.occupancy[end_r, end_c] = self.OCC_COLOR

            self.floor.tiles[pixel[0],pixel[1]] = pixel_color
        
        self.floor.save_image("slam_floor.png", "occupancy_grid.png")

    def get_N8_unknown_neighbours(self, point):
        neighbours = []
        r, c = point
        directions = [[-1, -1], [0,-1], [1, -1],
                      [-1, 0],          [1, 0],
                      [-1, 1],  [0, 1], [1, 1]]
        
        for x, y in directions:
            nx = r + x
            ny = c + y 
            if self.in_bounds((nx, ny)):
                considered_neighbour = self.floor.occupancy[nx, ny]
                if np.array_equal(considered_neighbour, self.UNKNOWN_COLOR):
                    neighbours.append((nx, ny))

        return neighbours
    
    def get_frontier_cells(self):
        frontier_cells = []

        for r in range(self.height):
            for c in range(self.width):
                if np.array_equal(self.floor.occupancy[r, c], self.FREE_COLOR):
                    neighbours = self.get_N8_unknown_neighbours((r,c))

                    if (len(neighbours) > 0):
                        frontier_cells.append((r,c))

        return np.array(frontier_cells, dtype = int)


    def get_target_frontier_cell(self, roomba, visited_frontiers=None):
        self.frontier_cells = self.get_frontier_cells()
        if self.frontier_cells.size == 0:
            return None

        roomba_pixel = self.get_roomba_pixel_position(roomba)
        dists = np.linalg.norm(self.frontier_cells - roomba_pixel, axis=1)

        min_dist = 1.0  
        valid_mask = dists > min_dist

        if visited_frontiers is not None and len(visited_frontiers) > 0:
            not_visited = np.array(
                [tuple(rc) not in visited_frontiers for rc in self.frontier_cells]
            )
            valid_mask &= not_visited

        if not np.any(valid_mask):
            return None

        filtered_frontiers = self.frontier_cells[valid_mask]
        filtered_dists = dists[valid_mask]

        closest_idx = np.argmin(filtered_dists)
        target_frontier = filtered_frontiers[closest_idx]
        return target_frontier

    def cluster_frontiers(self, frontier_cells):
        frontier_cells = np.asarray(frontier_cells, dtype=int)
        frontier_set = set(map(tuple, frontier_cells))
        visited = set()
        clusters = []

        directions = [(-1, -1), (0, -1), (1, -1),
                      (-1,  0),          (1,  0),
                      (-1,  1), (0,  1), (1,  1)]

        for cell in frontier_cells:
            cell = tuple(cell)
            if cell in visited:
                continue

            queue = [cell]
            visited.add(cell)
            cluster = [cell]

            while queue:
                r, c = queue.pop()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    nbr = (nr, nc)
                    if nbr in frontier_set and nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
                        cluster.append(nbr)

            clusters.append(np.array(cluster, dtype=int))

        return clusters

    def get_cluster_goal(self, roomba, visited_goals=None):
        frontier_cells = self.get_frontier_cells()
        if frontier_cells.size == 0:
            print("Frontier cells == 0 ")
            return None

        clusters = self.cluster_frontiers(frontier_cells)
        print("Number of Clusters: ", len(clusters))
        if not clusters:
            print("not Clusters")
            return None

        roomba_pixel = self.get_roomba_pixel_position(roomba)

        # if visited_goals is None:
        #     visited_goals = set()

        best_score = -np.inf
        best_centroid = None

        for cluster in clusters:
            # centroid in grid coordinates
            mean_rc = cluster.mean(axis=0)
            d = np.linalg.norm(cluster - mean_rc, axis=1)
            centroid_rc = cluster[np.argmin(d)]
            centroid_t = tuple(centroid_rc)

            # skip clusters we've already used as goals
            # if centroid_t in visited_goals:
            #     continue

            size = cluster.shape[0]
            dist = np.linalg.norm(centroid_rc - roomba_pixel)

            # tiny epsilon to avoid div by zero
            score = size / (dist + 1e-3)

            print ("score: ", score, " best_score: ", best_score)
            if score > best_score:
                best_score = score
                best_centroid = centroid_rc

        if best_centroid is None:
            print("No best centroid")
            return None

        return best_centroid

    def bresenham_line(self, start, end):
        r0, c0 = int(start[0]), int(start[1])
        r1, c1 = int(end[0]), int(end[1])
        points = []

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1

        err = dr - dc
        while True:
            points.append((r0, c0))
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc
        return points
    
    def is_free(self, rc):
        r, c = rc
        return self.in_bounds((r, c)) and np.array_equal(self.floor.occupancy[r, c], self.FREE_COLOR)
    
    def man_h(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def astar_path(self, start_rc, goal_rc):
        start = (int(start_rc[0]), int(start_rc[1]))
        goal = (int(goal_rc[0]), int(goal_rc[1]))

        if not self.is_free(start) or not self.is_free(goal):
            return None
        
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        open_heap = []
        heapq.heappush(open_heap, (self.man_h(start, goal), 0, start))

        came_from = {}
        gscore = {start: 0}
        closed = set()

        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                # reconstruct
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path

            closed.add(cur)

            for dr, dc in nbrs:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not self.is_free(nxt):
                    continue
                new_g = g + 1
                if nxt not in gscore or new_g < gscore[nxt]:
                    gscore[nxt] = new_g
                    came_from[nxt] = cur
                    f = new_g + self.man_h(nxt, goal)
                    heapq.heappush(open_heap, (f, new_g, nxt))

        return None
    
    def path_to_world_waypoints(self, path_rc, step=1):
        if path_rc is None or len(path_rc) == 0:
            return []

        pts = []
        for i in range(0, len(path_rc), step):
            r, c = path_rc[i]
            xy = self.convert_pixel_to_point((r, c))[:2]
            pts.append(np.array([xy[0], xy[1]], dtype=float))

        return pts
    
    def convert_pixel_to_point(self, rc):
        r, c = rc
        # undo the center shift
        rc_centered = np.array([r, c]) - self.center
        x = rc_centered[0] * self.floor.tile_width
        y = rc_centered[1] * self.floor.tile_length
        return np.array([x, y], dtype=float)

    def convert_point_to_pixel(self, point):
        pixel = np.rint(point / np.array([self.floor.tile_width, self.floor.tile_length, 1]))
        return np.int32(pixel[:2])
    
class FrontierExplorationController(RoombaController):
    def __init__(self):
        # mapper has floor, occupancy, handle_snapshot, frontier logic
        self.mapper = ManualSlamController()
        self.traj = None
        self.exploration_done = False
        self.just_started = True
        self.visited_goal_angles = {}
        self.rotate_step_deg = 60     # increment each revisit
        self.rotate_init_deg = 90     # first revisit rotation
        self.rotate_max_deg = 720     # optional cap; or just use % 360
        self.pending_rotate_deg = 0  

        self.rotating = False
        self.rotate_remaining = 0.0  # seconds left to rotate
        self.rotate_dir = 1    

        self.goal_rc = None                 # current planned goal (grid rc)
        self.rotate_on_arrival = False 

    def start_rotate(self, degrees=90):
        # turn_left / turn_right sets angular vel to +/-1 rad/s (see Roomba.turn_left) :contentReference[oaicite:1]{index=1}
        radians = np.deg2rad(degrees)
        omega = 1.0  # rad/s, matches turn_left/turn_right
        self.rotate_remaining = radians / omega
        self.rotating = True

    def next(self, roomba, timestep):
        if self.exploration_done:
            roomba.stop()
            return
        
        # If we're in a forced rotation phase, rotate and wait it out.
        if self.rotating:
            if self.rotate_dir > 0:
                roomba.turn_left()
            else:
                roomba.turn_right()

            self.rotate_remaining -= timestep
            if self.rotate_remaining <= 0:
                self.rotating = False
                roomba.stop()  # stop so the next snapshot isn't mid-spin
            return

        # If we have no trajectory or we just finished one, update map and replan.
        if self.traj is None or self.traj.completed:

            if self.traj is not None and self.traj.completed:
                if self.rotate_on_arrival:
                    goal_t = tuple(int(x) for x in self.goal_rc)

                    deg = int(self.pending_rotate_deg)
                    print(f"Arrived at revisited goal {list(self.goal_rc)} -> rotating {deg} deg now")

                    # alternate spin direction to avoid bias
                    self.start_rotate(deg)

                    # increment the next revisit angle for this centroid
                    if goal_t in self.visited_goal_angles:
                        # nxt = self.visited_goal_angles[goal_t] + self.rotate_step_deg
                        nxt = self.visited_goal_angles[goal_t]
                        # keep it cycling in a sane range
                        # nxt = nxt % 360
                        if nxt == 0:
                            nxt = self.rotate_step_deg
                        self.visited_goal_angles[goal_t] = nxt

                    self.rotate_on_arrival = False
                    self.pending_rotate_deg = 0
                    return
        
            roomba.take_snapshot()
            self.mapper.handle_snapshot(roomba)

            target_rc = self.mapper.get_cluster_goal(roomba)
            if target_rc is None:
                print("Exploration finished: no frontier cells left.")
                self.exploration_done = True
                roomba.stop()
                return
            
            target_t = tuple(int(x) for x in target_rc)

            if target_t in self.visited_goal_angles:
                self.rotate_on_arrival = True
                self.pending_rotate_deg = self.visited_goal_angles[target_t]
                print(f"Revisiting centroid {list(target_rc)} -> will rotate {self.pending_rotate_deg} deg upon arrival")
            else:
                self.rotate_on_arrival = False
                self.pending_rotate_deg = 0
            
            start_rc = self.mapper.get_roomba_pixel_position(roomba)

            path_rc = self.mapper.astar_path(start_rc, target_rc)
            if path_rc is None:
                # Can't reach this cluster centroid with current free space.
                # Mark this goal as visited so we don't keep retrying it forever.
                # self.visited_goals.add(tuple(target_rc))
                print(f"Skipping unreachable goal {target_rc}")
                if target_t not in self.visited_goal_angles:
                    self.visited_goal_angles[target_t] = self.rotate_init_deg
                return
            
            self.goal_rc = target_rc

            if target_t not in self.visited_goal_angles:
                self.visited_goal_angles[target_t] = self.rotate_init_deg
                
            waypoints = self.mapper.path_to_world_waypoints(path_rc, step=1)

            print(f"New target frontier at grid {target_rc}, path len {len(path_rc)}, waypoints {len(waypoints)}")
            self.traj = TrajectoryController(waypoints)
            return 

        self.traj.next(roomba, timestep)