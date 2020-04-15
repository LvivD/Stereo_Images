import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2

class depth_map:
    def __init__(self, img_left, img_right):
        self.shape = img_left.shape
        
        if len(img_left.shape) == 2:
            self.shape.append(1)

        self.img_left = img_left
        self.img_right = img_right
        self.shifted_maps = {}
        self.mass_cooficients_maps = {}
        
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
        
        if cache and step in self.shifted_maps:
            if prt:
                plt.imshow(self.shifted_maps[step])
            return self.shifted_maps[step]
        
        shifted = self.shift(step)
        shift_mask = []

        shift_correct_obj = np.zeros(self.shape[:2], dtype='uint16')
        shift_correct_obj_one_color = np.zeros(self.shape[:2], dtype='uint16')

        for i in range(self.shape[2]):
            shift_mask = shifted[:,:,i] <= 7
            shift_correct_obj_one_color[shift_mask] = 256/4
            shift_correct_obj += shift_correct_obj_one_color
            
        if prt:
            plt.imshow(shift_correct_obj)
            
        if save:
            self.shifted_maps[step] = shift_correct_obj
            
        return shift_correct_obj
       
        
    def get_shifted_maps(self, steps, save=True, cache=True):
        
        if type(steps) == int:
            steps = range(steps)
        
        res = {}
        
        for step in steps:
        
            current_shifted_map = self.get_shifted_map(step, save=save, cache=cache)
            
            res[step] = current_shifted_map
            if save:
                self.shifted_maps[step] = current_shifted_map
            
        return res
    
#     @staticmethod
#     def default_mass_cooficients_matrix(m, k=2):
#         res = np.zeros((m,m), dtype=int)
#         for i in range(-k, k+1):
#             res += np.diag([1 for j in range(m-abs(i))], k=i)
#         return res
        
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
    
    def get_mass_cooficients_map(self, step, save=True, cache=True, prt=False):
        
        if cache and step in self.mass_cooficients_maps:
            if prt:
                plt.imshow(self.mass_cooficients_maps[step])
            return self.mass_cooficients_maps[step]
        
        mass_cooficients_matrix = np.ones((5,5),np.float32)/25
        mass_cooficients_map = cv2.filter2D(self.get_shifted_map(step),-1,mass_cooficients_matrix)
        
        mass_cooficients_map = mass_cooficients_map
        
        if prt:
            plt.imshow(mass_cooficients_map)
            
        if save:
            self.mass_cooficients_maps[step] = mass_cooficients_map
            
        return mass_cooficients_map
    
    def get_mass_cooficients_maps(self, steps, save=True, cache=True, prt=False):
        
        if type(steps) == int:
            steps = range(steps)
        
        res = {}
        
        for step in steps:
        
            current_mass_cooficients_map = self.get_mass_cooficients_map(step, save=save, cache=cache)
            
            res[step] = current_mass_cooficients_map
            if save:
                self.mass_cooficients_maps[step] = current_mass_cooficients_map
            
        return res
        