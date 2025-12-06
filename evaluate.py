import cv2
from prm_planner import PRMPlanner
import numpy as np
from skimage.metrics import structural_similarity as ssim

FLOOR_PATH = "floor.png"
SLAM_FLOOR_PATH = "slam_floor.png"
FREE = 255 * 3

def tiles_to_free_mask(tiles, target_type = int):
    return (tiles.sum(axis = -1) == FREE).astype(target_type)

# simplist similarity
# https://en.wikipedia.org/wiki/Jaccard_index
def get_iou_similarity(a, b):
    a_free = tiles_to_free_mask(a, bool)
    b_free = tiles_to_free_mask(b, bool)

    intersection = np.logical_and(a_free, b_free)
    union = np.logical_or(a_free, b_free)

    return intersection.sum() / union.sum()

# ssim 
# https://scikit-image.org/docs/0.25.x/auto_examples/transform/plot_ssim.html
def get_structural_similarity(a, b):
    score, _diff = ssim(
        a, 
        b, 
        win_size = 3,
        channel_axis = 2, # by defualt treated as grayscale
        full=True)
    return score

# a is the ground truth 
def get_f1_score(true_tiles, pred_tiles):
    true_mask = tiles_to_free_mask(true_tiles, bool)
    pred_mask = tiles_to_free_mask(pred_tiles, bool)

    TP = np.logical_and(true_mask, pred_mask).sum()
    FP = np.logical_and(np.logical_not(true_mask), pred_mask).sum()
    FN = np.logical_and(true_mask, np.logical_not(pred_mask)).sum()
    TN = np.logical_and(np.logical_not(true_mask), np.logical_not(pred_mask)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1 


def main():
    tiles = cv2.cvtColor(cv2.imread(FLOOR_PATH, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    slam_tiles = cv2.cvtColor(cv2.imread(SLAM_FLOOR_PATH, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    iou_similarity = get_iou_similarity(tiles, slam_tiles)
    print(f"iou_similarity = {iou_similarity}")

    structural_similarity = get_structural_similarity(tiles, slam_tiles)
    print(f"structural_similarity = {structural_similarity}")

    f1_score = get_f1_score(tiles, slam_tiles)
    print(f"f1_similarity = {f1_score}")



if __name__ == "__main__":
    main()
