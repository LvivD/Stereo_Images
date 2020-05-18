import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2


class DepthMap:
    def __init__(self, img_left: , img_right):

        if img_left.shape != img_right:
            raise ValueError("The sizes of images should coinside.")

        self.shape = img_left.shape

        self.max_step = 0

        if len(img_left.shape) == 2:
            self.shape.append(1)

        self.img_left = img_left
        self.img_right = img_right
        self.shifted_maps = []
        self.mass_cooficients_maps = []

    def shifted_photo(self, depth_map: np.ndarray, photo="l") -> np.ndarray:
        if photo == "l":
            working_image = self.img_left
        elif photo == "r":
            working_image = self.img_right
        else:
            raise ValueError("The value of photo parrameter should be 'r' or 'l'.")
        
        res_image = np.zeros(self.shape, dtype=int)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                try:
                    res_image[x,y] = working_image[x + depth_map[x, y] if x + depth_map[x, y] < self.shape[0] else self.shape[0] - 1,y]
                except IndexError:
                    print(depth_map[x, y], type(depth_map[x, y]))
                    raise IndexError

        return res_image

                

    def energy_function_absolute(self, depth_map: np.ndarray) -> float:

        e_data = np.sum(np.absolute(self.shifted_photo(depth_map) - self.img_right))
        e_smooth = 0

        return e_data + e_smooth

    def energy_function_square(self, depth_map: np.ndarray) -> float:

        e_data = np.sum((self.shifted_photo(depth_map) - self.img_right)**2)
        e_smooth = 0
        
        return e_data + e_smooth







