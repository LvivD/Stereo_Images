import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import numpy as np
from depth_map import DepthMap


class Graph:
    def __init__(self, edges=[]):
        self.graph = nx.Graph()
        self.pixel_matrix = np.zeros(0)
        self.terminal_nodes = []
        if edges:
            self.createGraph_deprecated(edges)

    def _neighbours_helper(self, x, y):
        values_of_v, sum_of_v = [], 0
        for neighbor in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            try:
                values_of_v.append(
                    V(self.pixel_matrix[x, y],
                        self.pixel_matrix[x + neighbor[0], y + neighbor[1]])
                )
                sum_of_v += values_of_v[len(values_of_v - 1)]
            except IndexError:
                values_of_v.append(None)
        return sum_of_v, values_of_v

    def createGraph(self, depth_map: DepthMap):

        self.pixel_matrix = np.zeros(depth_map.shape, dtype=Node)
        self.terminal_nodes = [Terminal(i) for i in range(depth_map.max_step)]

        for x in range(depth_map.shape[0]):
            for y in range(depth_map.shape[1]):
                self.pixel_matrix[x, y] = Pixel(
                    (x, y), depth_map.depth_map[x, y], (depth_map.img_left[x, y], depth_map.img_right[x, y]))

        for x in range(self.pixel_matrix.shape[0]):
            for y in range(self.pixel_matrix.shape[1]):

                sum_of_v, values_of_v = self._neighbours_helper(x, y)

                try:
                    2*values_of_v[1]
                    self.addEdge(
                        self.pixel_matrix[x, y],
                        self.pixel_matrix[x+1, y],
                        V(self.pixel_matrix[x, y], self.pixel_matrix[x+1, y])
                    )
                except TypeError:
                    pass
                try:
                    2*values_of_v[2]
                    self.addEdge(
                        self.pixel_matrix[x, y],
                        self.pixel_matrix[x, y+1],
                        V(self.pixel_matrix[x, y], self.pixel_matrix[x, y+1]))
                except TypeError:
                    pass

                self.addEdge(
                    self.pixel_matrix[x, y],
                    self.terminal_nodes[self.pixel_matrix[x, y].lable],
                    D(self.pixel_matrix[x, y], self.pixel_matrix[x +
                                                                 self.pixel_matrix[x, y].lable]) + sum_of_v
                )

    def createGraph_deprecated(self, edges):
        self.addEdges(edges)

    def addEdges(self, edges: tuple or list):
        for edge in edges:
            self.addEdge(edge[0], edge[1], edge[2])

    def addEdge(self, nodeA, nodeB, weight):
        self.graph.add_edge(nodeA, nodeB, weight=weight)

    def optimize(self):
        x, y = 0, 0
        # while x < self.depth_map.shape[0]:
        #     while y < self.depth_map.shape[1]:
        #         sum_of_v, values_of_v = self._neighbours_helper(x, y)
        #         for neighbour in ((1, 0), (0, -1)):
        #             pass

    def cut(self):
        cut_value, partition = nx.stoer_wagner(self.graph)
        cutset = set()
        index = 0 if (len(partition[0]) < len(partition[1])) else 1
        for u in partition[index]:
            for v in partition[(index+1) % 2]:
                try:
                    cutset.add((u, v, self.graph[u][v]['weight']))
                except KeyError:
                    pass
        return cut_value, cutset


class Node:
    def __init__(self, label: int):
        self.label = label


class Terminal(Node):
    def __init__(self, label: int, connected_with=set()):
        super().__init__(label)
        self.connected_with = connected_with


class Pixel(Node):
    def __init__(self, coor: list or tuple, label: int, intensity: tuple or list):
        super().__init__(label)
        self.coor = coor
        self.intensity = intensity  # (left, right)
        self.lable = label


def D(p: Pixel, q: Pixel):
    # TODO help needed
    # p -> pixel from left img (steady); q -> p's image from right img (movable)
    return np.mean(np.absolute(p.intensity[0] - q.intensity[1]))


def V(p: Pixel, q: Pixel, K=80, sigma=5):
    return K*max(sigma, abs(p.lable - q.lable))

# def treminalWeight(p: Node, labeling_ab, neighbours):
#     summa = D(p.l)
#     for q in neighbours:
#         if q not in labeling_ab:
#             summa += V(p, q)


def edgeWeight(labelA, labelB):
    return V(labelA, labelB)


if __name__ == "__main__":
    edges = [('x', 'a', 3.0), ('x', 'b', 1.0), ('a', 'c', 3.0),
             ('b', 'c', 5.0), ('b', 'd', 4.0), ('d', 'e', 2.0), ('c', 'y', 2.0), ('e', 'y', 3.0)]
    G = Graph(edges)
    print(G.cut())
