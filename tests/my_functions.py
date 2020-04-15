import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def shift(photo, step):
    base_photo = photo[0].copy()
    shifted_photo = photo[1].copy()
    
    shifted_photo[:, [i+step for i in range(photo[1].shape[1] - step)]] = shifted_photo[:, [i for i in range(photo[1].shape[1] - step)]]
    
    shifted_photo = shifted_photo - base_photo
    shifted_photo[:, :step] = 256/2
    
    return shifted_photo

def mass_function1(shift_img, val = 10):
    res = 0
    for i in shift_img.ravel():
        if i < val:
            res += 1
    return res

def mass_function2(shift_img, val = 10):
    res = 0
    for x in range(shift_img.shape[0]):
        for y in range(shift_img.shape[1]):
            for z in range(shift_img.shape[2]):
                if shift_img[x, y, z] < val:
                    res += 1
                    for nearby in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        try:
                            if shift_img[x + nearby[0], y + nearby[1], z] < val:
                                res += 1
                        except IndexError:
                            pass
    return res

def save_resaults(img, steps, path):
    
    if type(steps) == int:
        steps = range(steps)
    
    for step in steps:
        shift2 = shift(img, step)
        shift2_mask = []

        shift2_correct_obj = shift2.copy()
        shift2_correct_obj[:,:] = 0
        shift2_correct_obj_one_color = shift2.copy()
        shift2_correct_obj_one_color[:,:] = 0
        
        for i in range(3):
            shift2_mask = shift2[:,:,i] <= 7
            shift2_correct_obj_one_color[shift2_mask] = [256/4, 256/4, 256/4]
            shift2_correct_obj += shift2_correct_obj_one_color

#         shift2_correct_obj = shift2.copy()
#         shift2_correct_obj[:,:] = 0

#         for i in range(3):
#             shift2_mask = shift2[:,:,i] <= 7
#             shift2_correct_obj[shift2_mask] = [256*3/4, 256*3/4, 256*3/4]

        plt.imshow(shift2_correct_obj)
        plt.savefig(path + str(step) + ".png")
    return

def get_shifted_maps(img, steps):
    
    if type(steps) == int:
        steps = range(steps)
    
    res = [[] for i in steps]
    
    for step in steps:
        shift2 = shift(img, step)
        shift2_mask = []

        shift2_correct_obj = np.zeros(shift2.shape[:2], dtype='uint16')
        shift2_correct_obj_one_color = np.zeros(shift2.shape[:2], dtype='uint16')
        
        
        
        for i in range(3):
            shift2_mask = shift2[:,:,i] <= 7
            shift2_correct_obj_one_color[shift2_mask] = 256/4
            shift2_correct_obj += shift2_correct_obj_one_color

#         shift2_correct_obj = shift2.copy()
#         shift2_correct_obj[:,:] = 0

#         for i in range(3):
#             shift2_mask = shift2[:,:,i] <= 7
#             shift2_correct_obj[shift2_mask] = [256*3/4, 256*3/4, 256*3/4]

        plt.imshow(shift2_correct_obj)
        res[step] = shift2_correct_obj
    return res

