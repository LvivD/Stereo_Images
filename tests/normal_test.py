from skimage import io
from depth_map import DepthMap
import numpy as np
import matplotlib.pyplot as plt

from graph import Graph


img = []
for i in [1, 5]:
    img.append(
        io.imread("datasets/scenes2001/tsukuba/scene1.row3.col" + str(i) + ".ppm"))

# img = []
# for i in [1,5]:
#     img.append(io.imread("../datasets/scenes2001/tsukuba/scene1.row3.col" + str(i) + ".ppm"))


dm = DepthMap(img[0], img[1])

a = dm.shift(21, prt=True)
a = ((a//255)**2)*255

dm.get_shifted_maps(60)

dm.get_mass_cooficients_maps(60)

mass_cooficients_matrix = np.matrix([[0, 0, 0, 0.05, 0.1, 0.2, 0.1, 0.05, 0, 0, 0],
                                     [0, 0, 0.05, 0.1, 0.25, 0.4,
                                         0.25, 0.1, 0.05, 0, 0],
                                     [0, 0.05, 0.1, 0.25, 0.55, 0.6,
                                         0.55, 0.25, 0.1, 0.05, 0],
                                     [0.05, 0.1, 0.25, 0.55, 0.75, 1,
                                         0.75, 0.55, 0.25, 0.1, 0.05],
                                     [0.1, 0.25, 0.55, 0.75, 1, 1,
                                         1, 0.75, 0.55, 0.25, 0.1],
                                     [0.2, 0.4, 0.6, 1, 1, 1, 1, 1, 0.6, 0.4, 0.2],
                                     [0.1, 0.25, 0.55, 0.75, 1, 1,
                                         1, 0.75, 0.55, 0.25, 0.1],
                                     [0.05, 0.1, 0.25, 0.55, 0.75, 1,
                                         0.75, 0.55, 0.25, 0.1, 0.05],
                                     [0, 0.05, 0.1, 0.25, 0.55, 0.6,
                                         0.55, 0.25, 0.1, 0.05, 0],
                                     [0, 0, 0.05, 0.1, 0.25, 0.4,
                                         0.25, 0.1, 0.05, 0, 0],
                                     [0, 0, 0, 0.05, 0.1, 0.2, 0.1, 0.05, 0, 0, 0]])/37.2

dm.change_axis_in_mcm()

res = dm.get_depth_map()


G = Graph()
G.createGraph(dm)
