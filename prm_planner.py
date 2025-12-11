from networkx.exception import NetworkXNoPath
from prm_util import *

class PRMPlanner():

    """
    args:
        tiles: tiles from a FLoor object 
        connectivity_radius: radius for local planner connectivity
        num_samples: number of samples; note that samples in obstacle will be discarded
    """
    def __init__(
            self, 
            tiles,
            connectivity_radius = 10,
            num_samples = 1000):

        # convert into mask of free space 
        # empty spaces are white (255, 255, 255) 
        # tiles_free is H * W of {0, 1}
        FREE = 255 * 3
        tiles_free = (tiles.sum(axis = -1) == FREE).astype(int)

        # --- samples points ---
        h = tiles_free.shape[0]
        w = tiles_free.shape[1]
        # samples = sample_naive_random(h, w, num)
        samples = sample_grid_w_offset(h, w, num_samples)
        # samples = sample_halton(h, w, num)

        # --- add nodes and edges to graph --- 
        map, kdtree = create_map_kdtree(samples, tiles_free, connectivity_radius)

        self.tiles_free = tiles_free
        self.samples = samples
        self.connectivity_radius = connectivity_radius
        self.map = map
        self.kdtree = kdtree
        

    """
    args:
        source: (x, y) point 
        goal: (x, y) point
    returns:
        path: list of (x, y) points including source and goal
    """
    def plan(self, source, goal):
        source_goal = [source, goal]
        # if points already in graph, don't change
        # otherwise add to map and remove after done
        should_remove_after = [False, False]
        for i in range(len(source_goal)):
            point = source_goal[i]
            if not self.map.has_node(point):
                add_edges_to_map(
                    self.map,
                    self.samples,
                    self.kdtree, 
                    point, 
                    self.connectivity_radius,
                    self.tiles_free)
                should_remove_after[i] = True
        # now find shortest distance 
        try:
            path = shortest_path(self.map, source ,goal)
        except NetworkXNoPath:
            path = None
        finally:
            # clean up 
            for i in range(len(source_goal)):
                if should_remove_after[i]:
                    self.map.remove_node(source_goal[i])
        return path

    """
    visualize the planner
    args:
        path: the path from plan(); if empty, no path is drawn 
    """
    def visualize_prm(self, path = None) -> None:
        img = mask_to_showable_image(self.tiles_free)
        for pt_a, pt_b in self.map.edges():
            cv2.line(img, pt_a, pt_b, color=CV2_LIGHT_BLUE, thickness=1)
            cv2.circle(img, pt_a, 0, CV2_LIGHT_PINK, -1)
            cv2.circle(img, pt_b, 0, CV2_LIGHT_PINK, -1)

        # visalize path
        if path is not None:
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

