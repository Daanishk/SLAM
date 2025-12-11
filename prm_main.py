from floor import Floor
import cv2
from prm_planner import PRMPlanner

def main():
    # load raw floorplan 
    # whic is ndarray of shape (H, W, 3)
    floor = Floor(1, 1, 5)
    floor.set_image("floor.png")
    tiles = floor.tiles

    # resizing the image to 300 by 300 
    tiles_large = cv2.resize(
        tiles,
        (tiles.shape[0] * 10, tiles.shape[1]*10)
    )

    # create planner
    prm_planner = PRMPlanner(
        tiles_large,
        10, 
        2000
    )

    # some example points in the map
    points = [(43, 65), (61, 97), (56, 188), # blue room 
              (132, 226), (141, 159), (127, 55), # black corridoor
              (190, 191), (193, 211), # green room
              (218, 63), (198, 130) ] # red room 
    outside_points = [(0, 0), (300, 300)]

    # query planner
    source = points[7]
    goal = outside_points[0]
    path = prm_planner.plan(source, goal)
    
    prm_planner.visualize_prm(path)    

if __name__ == "__main__":
    main()
