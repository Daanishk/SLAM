import cv2
from prm_planner import PRMPlanner
import numpy as np

FLOOR_PATH = "floor.png"
SLAM_FLOOR_PATH = "slam_floor.png"
FREE = 255 * 3

def tiles_to_free_mask(tiles, target_type = int):
    return (tiles.sum(axis = -1) == FREE).astype(target_type)

def get_iou_similarity(a, b):
    a_free = tiles_to_free_mask(a, bool)
    b_free = tiles_to_free_mask(b, bool)

    intersection = np.logical_and(a_free, b_free)
    union = np.logical_or(a_free, b_free)

    return intersection.sum() / union.sum()


def main():
    tiles = cv2.cvtColor(cv2.imread(FLOOR_PATH, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    slam_tiles = cv2.cvtColor(cv2.imread(SLAM_FLOOR_PATH, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    iou_similarity = get_iou_similarity(tiles, slam_tiles)
    print(f"iou_similarity = {iou_similarity}")

    


if __name__ == "__main__":
    main()
