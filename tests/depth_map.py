import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2


class DepthMap:
    def __init__(self, img_left, img_right):
        self.shape = img_left.shape

        self.max_step = 0

        if len(img_left.shape) == 2:
            self.shape = (img_left.shape[0],img_left.shape[1],1)

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
            raise ValueError(
                "The value of photo parrameter should be 'r' or 'l'.")

        res_image = np.zeros(self.shape, dtype=int)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                try:
                    res_image[x, y] = working_image[x + depth_map[x, y] if x +
                                                    depth_map[x, y] < self.shape[0] else self.shape[0] - 1, y]
                except IndexError:
                    print(depth_map[x, y], type(depth_map[x, y]))
                    raise IndexError

        return res_image

    def energy_function_absolute(self, depth_map: np.ndarray) -> float:
        return np.sum(np.absolute(self.shifted_photo(depth_map) - self.img_right))

    def energy_function_square(self, depth_map: np.ndarray) -> float:
        return np.sum((self.shifted_photo(depth_map) - self.img_right)**2)

    def check_image_order(self):
        pass

    def plot(self, img='l'):
        if img == 'l':
            plt.imshow(self.img_left)
        elif img == 'r':
            plt.imshow(self.img_right)

    def shift(self, step, prt=False):

        shifted = self.img_right.copy()

        shifted[:, [i + step for i in range(self.shape[1] - step)]] = \
            shifted[:, [i for i in range(self.shape[1] - step)]]

        shifted = shifted - self.img_left
        shifted[:, :step] = 256/2

        if prt:
            plt.imshow(shifted)

        return shifted

    def get_shifted_map(self, step, save=True, cache=True, prt=False):

        if len(self.shifted_maps) <= step:
            self.shifted_maps += [None for i in range(
                step - len(self.shifted_maps) + 1)]

        if cache and not (self.shifted_maps[step] is None):
            if prt:
                plt.imshow(self.shifted_maps[step])
            return self.shifted_maps[step]

        shifted = self.shift(step)
        shift_mask = []

        shift_correct_obj = np.zeros(self.shape[:2], dtype='int16')
        shift_correct_obj_one_color = np.zeros(self.shape[:2], dtype='int16')

        for i in range(self.shape[2]):
            shift_mask = shifted[:, :, i] <= 7
            shift_correct_obj_one_color[shift_mask] = 256/4
            shift_correct_obj += shift_correct_obj_one_color

        if prt:
            plt.imshow(shift_correct_obj)

        if save:
            self.shifted_maps[step] = shift_correct_obj

        return shift_correct_obj

    def get_shifted_maps(self, steps, save=True, cache=True):

        res = [None for i in range(steps)]

        for step in range(steps):

            current_shifted_map = self.get_shifted_map(
                step, save=save, cache=cache)

            res[step] = current_shifted_map

        return res

    @staticmethod
    def default_mass_cooficients_matrix(m, k=2):
        res = np.zeros((m, m), dtype=int)
        for i in range(-k, k+1):
            res += np.diag([1 for j in range(m-abs(i))], k=i)
        return res

#     def get_mass_cooficients_map1(self, step, save=True, cache=True, prt=False):

#         if cache and step in self.mass_cooficients_maps:
#             if prt:
#                 plt.imshow(self.mass_cooficients_maps[step])
#             return self.mass_cooficients_maps[step]

#         mass_cooficients_map = np.matmul(np.matmul(self.default_mass_cooficients_matrix(self.shape[0]), self.get_shifted_map(step)), self.default_mass_cooficients_matrix(self.shape[1]))

#         mass_cooficients_map = mass_cooficients_map / 9

#         if prt:
#             plt.imshow(mass_cooficients_map)

#         if save:
#             self.mass_cooficients_maps[step] = mass_cooficients_map

#         return mass_cooficients_map

    def get_mass_cooficients_map(self, step, mass_cooficients_matrix=np.ones((5, 5), np.float32)/25, save=True, cache=True, prt=False):

        if len(self.mass_cooficients_maps) <= step:
            self.mass_cooficients_maps += [None for i in range(
                step - len(self.mass_cooficients_maps) + 1)]

        if cache and not (self.mass_cooficients_maps[step] is None):
            if prt:
                plt.imshow(self.mass_cooficients_maps[step])
            return self.mass_cooficients_maps[step]

        mass_cooficients_map = cv2.filter2D(
            self.get_shifted_map(step), -1, mass_cooficients_matrix)

        mass_cooficients_map = mass_cooficients_map

        if prt:
            plt.imshow(mass_cooficients_map)

        if save:
            self.mass_cooficients_maps[step] = mass_cooficients_map

        return mass_cooficients_map

    def get_mass_cooficients_maps(self, steps, mass_cooficients_matrix=np.ones((5, 5), np.float32)/25, save=True, cache=True, prt=False):

        if steps > self.max_step:
            self.max_step = steps

#         res = [None for i in range(steps)]
        res = np.zeros((steps, self.shape[0], self.shape[1]), dtype=int)

        for step in range(steps):

            current_mass_cooficients_map = self.get_mass_cooficients_map(
                step, mass_cooficients_matrix=mass_cooficients_matrix, save=save, cache=cache)

            res[step] = current_mass_cooficients_map

        return res

    def change_axis_in_mcm(self, mcm=None, save=False):
        if mcm is None:
            self.mass_cooficients_maps = np.swapaxes(
                self.mass_cooficients_maps, 0, 2)
            self.mass_cooficients_maps = np.swapaxes(
                self.mass_cooficients_maps, 0, 1)
            return self.mass_cooficients_maps
        else:
            mcm_new = mcm.copy()
            mcm_new = np.swapaxes(mcm_new, 0, 2)
            mcm_new = np.swapaxes(mcm_new, 0, 1)
            if save:
                self.mass_cooficients_maps = mcm_new
            return mcm_new

    def get_depth_map(self, mcm=None):
        if mcm is None:
            self.depth_map = np.zeros(self.shape[:2], dtype=int)
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    self.depth_map[x, y] = np.argmax(
                        self.mass_cooficients_maps[x, y])
            return self.depth_map
        else:
            depth_map = np.zeros(mcm.shape[:2])
            for x in range(mcm.shape[0]):
                for y in range(mcm.shape[1]):
                    depth_map[x, y] = np.argmax(mcm[x, y])
            return depth_map

    def erase_resaults(self):
        self.shifted_maps = []
        self.mass_cooficients_maps = []
        self.depth_map = []


#     def get_depth_map_particular_zones(self, k, max_step = None):

#         if max_step is None:
#             max_step = self.max_step//(2*k + 1)

#         self.depth_map = np.zeros(self.shape[:2])
#         depth_map_sum = self.mass_cooficients_maps.copy()
#         for x in range(self.mass_cooficients_maps.shape[0]):
#             for y in range(self.mass_cooficients_maps.shape[1]):
#                 for z in range(self.mass_cooficients_maps.shape[2]):
#                     j = 1
#                     for i in range(1, k+1):
#                         try:
#                             assert z-i >= 0
#                             depth_map_sum[x,y,z] += self.mass_cooficients_maps[x,y,z-i]
#                             j += 1
#                         except AssertionError:
#                             pass


#                         try:
#                             assert z+i < self.shape[2]
#                             depth_map_sum[x,y,z] += self.mass_cooficients_maps[x,y,z+i]
#                             j += 1
#                         except AssertionError:
#                             pass
#                     depth_map_sum[x,y,z] //= j


#         k_mass_cooficients_maps = np.zeros((self.shape[0], self.shape[1], max_step))

#         kk = self.mass_cooficients_maps.shape[2]//max_step

#         for x in range(self.shape[0]):
#             for y in range(self.shape[1]):
#                 for z in range(max_step):
#                     k_mass_cooficients_maps[x,y,z] = depth_map_sum[x,y,z*kk]


#         res = np.zeros(self.shape[:2])
#         for x in range(self.shape[0]):
#             for y in range(self.shape[1]):
#                 res[x,y] = np.argmax(k_mass_cooficients_maps[x,y]) + 1

#         return res


    def get_k_mass_cooficients_maps(self, k, steps, mass_cooficients_matrix=np.ones((5, 5), np.float32)/25, save=False, cache=True, prt=False):
        mass_cooficients_maps = self.get_mass_cooficients_maps(
            steps, mass_cooficients_matrix=mass_cooficients_matrix, save=save, cache=False, prt=False)

        print(mass_cooficients_maps.shape, DepthMap.default_mass_cooficients_matrix(
            mass_cooficients_maps.shape[0], 3).shape)

        mass_cooficients_maps = np.matmul(
            mass_cooficients_maps, DepthMap.default_mass_cooficients_matrix(
                mass_cooficients_maps.shape[0], 3)
        )//3

        mass_cooficients_maps = self.change_axis_in_mcm(
            mcm=mass_cooficients_maps)

        res = self.get_depth_map(mcm)

        return res
