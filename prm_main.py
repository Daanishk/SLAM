from floor import * 
from networkx import Graph, shortest_path
import time
from scipy.spatial import KDTree
from scipy.stats.qmc import Halton
import matplotlib.pyplot as plt

# cv2 uses gbr 
CV2_RED = (0, 0, 255)
CV2_LIGHT_PINK = (204, 153, 255)
CV2_BLUE = (255, 0, 0)
CV2_LIGHT_BLUE = (255, 204, 0)

# for (H, W, 3) 
def show_tiles(tiles: np.ndarray) -> None:
    cv2.imshow("tiles", tiles)
    cv2.waitKey(0)  # 0 for inf window life time
    cv2.destroyAllWindows()

# for (H, W)
# and each value is 0 / 1
def mask_to_showable_image(tiles: np.ndarray) -> np.ndarray:
    if tiles.ndim != 3:
        tiles = np.stack([tiles]*3, axis = -1)
    tiles = cv2.normalize(tiles, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return tiles

def visualize_prm(tiles_free: np.ndarray, map: Graph, path = []) -> None:
    img = mask_to_showable_image(tiles_free)
    for pt_a, pt_b in map.edges():
        cv2.line(img, pt_a, pt_b, color=CV2_LIGHT_BLUE, thickness=1)
        cv2.circle(img, pt_a, 0, CV2_LIGHT_PINK, -1)
        cv2.circle(img, pt_b, 0, CV2_LIGHT_PINK, -1)

    # visalize path
    for i in range(len(path) - 1):
        pt_a = path[i]
        pt_b = path[i + 1]
        cv2.line(img, pt_a, pt_b, color=CV2_BLUE, thickness=1)
        cv2.circle(img, pt_a, 0, CV2_RED, -1)
        cv2.circle(img, pt_b, 0, CV2_RED, -1)
    cv2.circle(img, path[0], 2, CV2_RED, -1)
    cv2.circle(img, path[-1], 2, CV2_RED, -1)

    # cv2.imshow("map", tile_w_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(cv2_img)
    plt.title('prm')
    plt.show()


# sample 2D points
# return np array of shape (num, 2)
def sample_naive_random(h: int, w: int, num: int) -> np.ndarray:
    np.random.seed(1984)
    sample_u = np.random.randint(low=0, high = h, size=num)
    sample_v = np.random.randint(low=0, high = w, size=num)
    samples = np.stack([sample_u, sample_v], axis=-1)
    return samples

# draw sample from a grid, offset for some randomness
def sample_grid_w_offset(h: int, w: int, num: int) -> np.ndarray:
    np.random.seed(1984)
    num_per_dim = int(np.sqrt(num))
    sample_u = np.linspace(0, w, num_per_dim, endpoint=False)
    sample_v = np.linspace(0, h, num_per_dim, endpoint=False)
    points = []
    cell_w = w / num_per_dim
    cell_h =  h / num_per_dim
    for u in sample_u:
        for v in sample_v:
            offset_u = np.random.uniform(0, cell_w / 1.5)
            offset_v = np.random.uniform(0, cell_h / 1.5)
            points.append((int(u + offset_u), int(v + offset_v)))
    return np.array(points)

# using halton sequence 
def sample_halton(h: int, w:int, num: int) -> np.ndarray:
    halton_sequence = Halton(d=2, scramble=True)
    halton_samples = halton_sequence.random(num)  # this is float in range [0, 1]
    halton_samples[:, 0] *= h 
    halton_samples[:, 1] *= w
    samples = halton_samples.astype(int)
    return samples

# check if two points shoule be connected
# this should be used if walls are always thicker than radius
def should_connect_ignoring_collsion(
        pt_a: tuple, 
        pt_b: tuple, 
        radius: float) -> bool:
    # don't self connect
    if pt_a == pt_b:
        return False
    # check within raiuds 
    distance = np.linalg.norm(np.array(pt_a) - np.array(pt_b))
    if distance > radius:
        return False
    return True

# this should be used when map is small and checking every pixel on path is viable
# this turned out to be pretty effecient by making use of matrix operaitons
# pass radius = -1 to skip radius check for use with dk tree 
def should_connect_exhaustive_check(
        pt_a: tuple, 
        pt_b: tuple, 
        radius: float, 
        tiles_free: np.ndarray) -> bool:
    # don't self connect
    if pt_a == pt_b:
        return False
    # check within raiuds 
    if radius >= 0:
        distance = np.linalg.norm(np.array(pt_a) - np.array(pt_b))
        if distance > radius:
            return False
    # check for collision
    mask = np.zeros(tiles_free.shape, dtype=np.uint8)
    # cv2.line essentially updates pixels on a line between points
    cv2.line(mask, pt_a, pt_b, color=1, thickness=1)
    collision = np.any((mask == 1) & (tiles_free == 0))
    if collision:
        return False
    return True

# don't use  
# this should be used when map is small and checking every pixel on path is viable
# this turns out to be horribly ineffecient due to manual loops
def _should_connect_w_interval(
        pt_a: tuple, 
        pt_b: tuple, 
        radius: float, 
        interval: float,
        tiles_free: np.ndarray) -> bool:
    # don't self connect
    if pt_a == pt_b:
        return False
    # check within raiuds 
    distance = np.linalg.norm(np.array(pt_a) - np.array(pt_b))
    if distance > radius ** 2:
        return False
    # check for collision
    a_to_b = np.array(pt_b) - np.array(pt_a)
    num_intervals = int(np.linalg.norm(a_to_b) / interval)
    a_to_b_unit = a_to_b / np.linalg.norm(a_to_b)
    tiles_free_T = tiles_free.transpose()
    for i in range(1, num_intervals + 1, 1):
        check_pt = np.floor(pt_a + i * a_to_b_unit * interval).astype(int)
        if tiles_free_T[tuple(check_pt)] == 0:
            return False
    return True

# add nodes and edges to tree using 
# naive edge construction with exhaustive search
def create_map_brute_force(
        samples: np.ndarray, 
        tiles_free: np.ndarray, 
        radius: float,
        local_planner = should_connect_exhaustive_check) -> Graph:
    # adding points in free space to graph
    # each point is a tuple: (u, v)
    map = Graph()
    for i in range(samples.shape[0]):
        point = tuple(samples[i])
        # don't add if not in free space
        # there is a transpose needed here
        tiles_free_T = tiles_free.transpose()
        if tiles_free_T[point] == 1:
            map.add_node(point)

    for pt_a in map.nodes:
        for pt_b in map.nodes:
            should_connect = local_planner(pt_a, pt_b, radius, tiles_free)
            if should_connect:
                weight = np.linalg.norm(np.array(pt_a) - np.array(pt_b))
                map.add_edge(pt_a, pt_b, weight=weight)
    return map

# add edges from given point to map 
# within radius, and using tiles_free to checke for obstacle
def add_edges_to_map(
        map: Graph,
        samples: np.ndarray, 
        kdtree: KDTree, 
        point: np.ndarray, 
        radius: float, 
        tiles_free: np.ndarray) -> None:
    points_in_radius = kdtree.query_ball_point(point, r=radius)
    neighbor_list = samples[points_in_radius]
    for i in range(neighbor_list.shape[0]):
        pt_b = tuple(neighbor_list[i])
        should_connect = should_connect_exhaustive_check(point, pt_b, -1, tiles_free)
        if should_connect:
            weight = np.linalg.norm(np.array(point) - np.array(pt_b))
            map.add_edge(point, pt_b, weight=weight)

# add nodes and edges to tree using 
# using kd tree
def create_map_kdtree(
        samples: np.ndarray, 
        tiles_free: np.ndarray, 
        radius: float,
        local_planner = should_connect_exhaustive_check) -> tuple[Graph, KDTree]:
    kdtree = KDTree(samples)
    map = Graph()
    for i in range(samples.shape[0]):
        point = tuple(samples[i])
        # there is a transpose needed here
        tiles_free_T = tiles_free.transpose()
        if tiles_free_T[point] == 1:
            map.add_node(point)

    for point in map.nodes:
        add_edges_to_map(
            map,
            samples, 
            kdtree, 
            point, 
            radius, 
            tiles_free)

    return map, kdtree

def main():
    # load raw floorplan 
    # whic is ndarray of shape (H, W, 3)
    floor = Floor(1, 1, 5)
    floor.set_image("floor.png")
    tiles = floor.tiles
    # show_tiles(tiles)

    tiles_large = cv2.resize(
        tiles,
        (tiles.shape[0] * 10, tiles.shape[1]*10)
    )

    # convert into mask of free space 
    # empty spaces are white (255, 255, 255) 
    # tiles_free is H * W of {0, 1}
    FREE = 255 * 3
    tiles_free = (tiles_large.sum(axis = -1) == FREE).astype(int)
    # show_mask(tiles_free)

    # --- samples points ---
    h = tiles_free.shape[0]
    w = tiles_free.shape[1]
    num = 3000
    # samples = sample_naive_random(h, w, num)
    samples = sample_grid_w_offset(h, w, num)
    # samples = sample_halton(h, w, num)

    # --- add nodes and edges to graph --- 
    radius = 10

    # brute force
    # start_time = time.time()
    # map = create_map_brute_force(samples, tiles_free, radius)
    # time_span = time.time() - start_time
    # print(f"map creation for {map.number_of_nodes()} nodes using brute force took {time_span:.3f} seconds")

    # improved edge construction with KD tree 
    start_time = time.time()
    map, kdtree = create_map_kdtree(samples, tiles_free, radius)
    time_span = time.time() - start_time
    print(f"map creation for {map.number_of_nodes()} nodes using kdtree took {time_span:.3f} seconds")

    # example of accessing: 
    # for pt_a, pt_b, weight in map.edges.data("weight"):
    #     print(f"{pt_a}, {pt_b}, {weight}")

    # visualize roadmap 
    # visualize_prm(tiles_free, map)

    # some sample points in the map
    points = [(43, 65), (61, 97), (56, 188), # blue room 
              (132, 226), (141, 159), (127, 55), # black corridoor
              (190, 191), (193, 211), # green room
              (218, 63), (198, 130) ] # red room 

    query = points[7]
    goal = points[9]
    query_goal = [query, goal]
    # if points already in graph, don't change
    # otherwise add to map and remove after done
    should_remove_after = [False, False]
    for i in range(len(query_goal)):
        point = query_goal[i]
        if not map.has_node(point):
            add_edges_to_map(
                map,
                samples,
                kdtree, 
                point, 
                radius,
                tiles_free)
            should_remove_after[i] = True
    # now find shortest distance 
    path = shortest_path(map, query ,goal)

    visualize_prm(tiles_free, map, path)    

if __name__ == "__main__":
    main()
