import numpy as np
import cv2

class Floor:
    def __init__(self, tile_width, tile_length, wall_height):
        self.tile_width = tile_width
        self.tile_length = tile_length
        self.wall_height = wall_height

    def init_tiles(self, width, height):
        self.tiles = np.ones((width,height,3),dtype=np.uint8) * 255

    def set_image(self, image_path):
        self.tiles = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def save_image(self, image_path):
        if (self.tiles is not None):
            cv2.imwrite(image_path, cv2.cvtColor(self.tiles, cv2.COLOR_RGB2BGR))

    def display_image(self, title="Floor"):
        if (self.tiles is not None):
            cv2.imshow(title,cv2.cvtColor(self.tiles, cv2.COLOR_RGB2BGR))

    def get_free_mask(self):
        # @TODO get mask of self.tiles that is free vs. obstacle or outside
        pass

    def similarity(self, other_floor):
        # @TODO: Needed for error calculation
        pass
