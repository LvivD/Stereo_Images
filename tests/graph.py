import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import numpy as np
from depth_map import DepthMap
from itertools import combinations


class Graph:
    my_inf = 100500

    def __init__(self, edges=[]):
        self.graph = nx.DiGraph()
        self.upper_pixel_matrix = np.zeros(0)
        self.lower_pixel_matrix = np.zeros(0)
        self.upper_terminal_nodes = []
        self.lower_terminal_nodes = []
        if edges:
            self.createGraph_deprecated(edges)

    def _count_sum_of_v(self, coords):
        x = coords[0]
        y = coords[1]
        sum_of_v = 0
        for neighbor in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            try:
                sum_of_v += V(self.upper_pixel_matrix[coords],
                              self.upper_pixel_matrix[x + neighbor[0], y + neighbor[1]])
            except IndexError:
                pass
        return sum_of_v

    def createGraph(self, depth_map: DepthMap):

        self.upper_pixel_matrix = np.zeros(depth_map.shape[:2], dtype=Node)
        self.lower_pixel_matrix = np.zeros(depth_map.shape[:2], dtype=Node)

        self.upper_terminal_nodes = [
            Terminal(i) for i in range(depth_map.max_step)]
        self.lower_terminal_nodes = [
            Terminal(i) for i in range(depth_map.max_step)]
        for i in range(len(self.upper_terminal_nodes)):
            self.addEdge(
                self.upper_terminal_nodes[i], self.lower_terminal_nodes[i], Graph.my_inf)


        for x in range(depth_map.shape[0]):
            for y in range(depth_map.shape[1]):
                self.upper_pixel_matrix[x, y] = Pixel(
                    (x, y), depth_map.depth_map[x, y], (depth_map.img_left[x, y], depth_map.img_right[x, y]))
                self.lower_pixel_matrix[x, y] = Lower(
                    depth_map.depth_map[x, y], self.upper_pixel_matrix[x, y])
                self.addEdge(
                    self.upper_pixel_matrix[x, y], self.lower_pixel_matrix[x, y], Graph.my_inf)
        
        print('step 0:', self.graph.number_of_edges())

        for x in range(self.upper_pixel_matrix.shape[0]):
            for y in range(self.upper_pixel_matrix.shape[1]):

                sum_of_v = self._count_sum_of_v((x, y))

                try:
                    edge_value = V(
                        self.upper_pixel_matrix[x, y], self.upper_pixel_matrix[x+1, y])
                    self.addEdge(
                        self.lower_pixel_matrix[x, y], self.upper_pixel_matrix[x+1, y], edge_value)
                    self.addEdge(
                        self.lower_pixel_matrix[x+1, y], self.upper_pixel_matrix[x, y], edge_value)
                except IndexError:
                    pass
                try:
                    edge_value = V(
                        self.upper_pixel_matrix[x, y], self.upper_pixel_matrix[x, y+1])
                    self.addEdge(
                        self.lower_pixel_matrix[x, y], self.upper_pixel_matrix[x, y+1], edge_value)
                    self.addEdge(
                        self.lower_pixel_matrix[x, y+1], self.upper_pixel_matrix[x, y], edge_value)
                except IndexError:
                    pass

                edge_value = D(
                    self.upper_pixel_matrix[x, y], self.upper_pixel_matrix[x - self.upper_pixel_matrix[x, y].label, y]) + sum_of_v
                self.addEdge(
                    self.lower_pixel_matrix[x, y],
                    self.upper_terminal_nodes[self.upper_pixel_matrix[x, y].label],
                    edge_value
                )
                self.addEdge(
                    self.lower_terminal_nodes[self.upper_pixel_matrix[x, y].label],
                    self.upper_pixel_matrix[x, y],
                    edge_value
                )
        print('graph number of edges =', self.graph.number_of_edges())
        print('graph number of nodes =', self.graph.number_of_nodes())

    def createGraph_deprecated(self, edges):
        self.addEdges(edges)

    def addEdges(self, edges: tuple or list):
        for edge in edges:
            self.addEdge(edge[0], edge[1], edge[2])

    def addEdge(self, nodeA, nodeB, weight):
        self.graph.add_edge(nodeA, nodeB, weight=weight)

    def optimize(self):

        # temp____________
        temp_added = 0
        temp_skiped = 0
        temp_pixels_with_different_labels = 0
        temp_lables_that_differ = set()
        # temp____________

        for x in range(self.upper_pixel_matrix.shape[0]):
            for y in range(self.upper_pixel_matrix.shape[1]):

                for neighbour in ((1, 0), (0, 1)):
                    this_node = (x, y)
                    neighbour_node = (x+neighbour[0], y+neighbour[1])
                    try:
                        

                        if self.upper_pixel_matrix[this_node].label != self.upper_pixel_matrix[neighbour_node].label:

                            # temp____________
                            temp_pixels_with_different_labels += 1
                            if self.upper_pixel_matrix[this_node].label <= self.upper_pixel_matrix[neighbour_node].label:
                                temp_lables_that_differ.add((self.upper_pixel_matrix[this_node].label, self.upper_pixel_matrix[neighbour_node].label))
                            else:
                                temp_lables_that_differ.add((self.upper_pixel_matrix[neighbour_node].label, self.upper_pixel_matrix[this_node].label))
                            # temp____________

                            # add possible_labels
                            self.upper_pixel_matrix[this_node].possible_labels.add(self.upper_pixel_matrix[neighbour_node].label)
                            self.upper_pixel_matrix[neighbour_node].possible_labels.add(self.upper_pixel_matrix[this_node].label)

                            edge_value = D(self.upper_pixel_matrix[this_node],
                                        self.upper_pixel_matrix[x - self.upper_pixel_matrix[neighbour_node].label, y]) + self._count_sum_of_v(this_node)
                            self.addEdge(
                                self.lower_pixel_matrix[this_node],
                                self.upper_terminal_nodes[self.upper_pixel_matrix[neighbour_node].label],
                                edge_value
                            )
                            self.addEdge(
                                self.lower_terminal_nodes[self.upper_pixel_matrix[neighbour_node].label],
                                self.upper_pixel_matrix[this_node],
                                edge_value
                            )

                            edge_value = D(self.upper_pixel_matrix[neighbour_node], self.upper_pixel_matrix[x -
                                                                                                            self.upper_pixel_matrix[this_node].label, y]) + self._count_sum_of_v(neighbour_node)
                            self.addEdge(
                                self.lower_pixel_matrix[neighbour_node],
                                self.upper_terminal_nodes[self.upper_pixel_matrix[this_node].label],
                                edge_value
                            )
                            self.addEdge(
                                self.lower_terminal_nodes[self.upper_pixel_matrix[this_node].label],
                                self.upper_pixel_matrix[neighbour_node],
                                edge_value
                            )
                        # temp____________
                        temp_added += 1
                        # temp____________
                    except IndexError:
                        # temp____________
                        temp_skiped += 1
                        # print(this_node)
                        # print(neighbour_node)
                        # temp____________
                        pass
        
        # temp____________
        print(temp_added, temp_skiped, temp_pixels_with_different_labels)
        print(len(temp_lables_that_differ))
        # temp____________
        
        for pair in combinations(range(len(self.upper_terminal_nodes)), 2): # pairs of all terminals nodes
            
            # cut the graph when sourse is upper node of first terminal node and sink is the lower node of terminal node
            if not nx.has_path(self.graph, self.lower_terminal_nodes[pair[0]], self.lower_terminal_nodes[pair[1]]):
                continue
            print('cut',pair)
            cut_value, cut_set = nx.minimum_cut(self.graph, self.lower_terminal_nodes[pair[0]], self.lower_terminal_nodes[pair[1]])
            # gets 2 sets where we should disconnect terminal nodes (upper and lower) from that set

            if (self.upper_terminal_nodes[pair[0]] in cut_set[0] and self.lower_terminal_nodes[pair[0]] in cut_set[0] and
                self.upper_terminal_nodes[pair[1]] in cut_set[1] and self.lower_terminal_nodes[pair[1]] in cut_set[1]):
                first_node_set = cut_set[0] - {self.upper_terminal_nodes[pair[0]], self.lower_terminal_nodes[pair[0]]}
                second_node_set = cut_set[1] - {self.upper_terminal_nodes[pair[1]], self.lower_terminal_nodes[pair[1]]}

            elif (self.upper_terminal_nodes[pair[0]] in cut_set[1] and self.lower_terminal_nodes[pair[0]] in cut_set[1] and 
                  self.upper_terminal_nodes[pair[1]] in cut_set[0] and self.lower_terminal_nodes[pair[1]] in cut_set[0]):
                first_node_set = cut_set[1] - {self.upper_terminal_nodes[pair[1]], self.lower_terminal_nodes[pair[1]]}
                second_node_set = cut_set[0] - {self.upper_terminal_nodes[pair[0]], self.lower_terminal_nodes[pair[0]]}

            else: 
                
                print(len(cut_set[0]), len(cut_set[1]))

                print(pair)


                print(self.upper_terminal_nodes[pair[0]] in cut_set[0])
                print(self.upper_terminal_nodes[pair[0]] in cut_set[1])
                print(self.lower_terminal_nodes[pair[0]] in cut_set[0])
                print(self.lower_terminal_nodes[pair[0]] in cut_set[1])

                print(self.upper_terminal_nodes[pair[1]] in cut_set[0])
                print(self.upper_terminal_nodes[pair[1]] in cut_set[1])
                print(self.lower_terminal_nodes[pair[1]] in cut_set[0])
                print(self.lower_terminal_nodes[pair[1]] in cut_set[1])



                raise RuntimeError("cut didn't work")

            # disconnects the sets from corresponding node and deletes wrong lables from possible_labels set
            for edge in first_node_set:
                # edge can be from upper or lower layer of nodes, so we need to check
                if type(edge) == Lower:
                    self.graph.remove_edge(edge ,self.upper_terminal_nodes[pair[0]])
                elif type(edge) == Pixel:
                    self.graph.remove_edge(self.lower_terminal_nodes[pair[0]], edge)
                    edge.possible_labels -= self.lower_terminal_nodes[pair[0]].label 
                
            
            for edge in second_node_set:
                self.graph.remove_edge(edge ,self.upper_terminal_nodes[pair[1]])
                self.graph.remove_edge(self.lower_terminal_nodes[pair[1]], edge)
            

            


        

        # cut_value, cut_set = self.cut()
        # for edge in cut_set:
        #     if isinstance(edge[0], Terminal):
        #         self.terminal_nodes[edge[0].label].remove(edge[0])
        #     elif isinstance(edge[1], Terminal):
        #         self.terminal_nodes[edge[1].label].remove(edge[1])

    def cut(self, TerminalA, TerminalB):
        cut_value, partition = nx.minimum_cut(self.graph, TerminalA, TerminalB)
        reachable, non_reachable = partition
        cutset = set()
        for u, nbrs in ((n, self.graph[n]) for n in reachable):
            cutset.update((u, v) for v in nbrs if v in non_reachable)
        return cut_value, cutset


class Node:
    def __init__(self, label: int):
        self.label = label


class Terminal(Node):
    def __init__(self, label: int, connected_with=set()):
        super().__init__(label)
        self.connected_with = connected_with


class Pixel(Node):
    def __init__(self, coor: list or tuple, label: int, intensity: tuple or list, possible_labels=set()):
        super().__init__(label)
        self.coor = coor
        self.intensity = intensity  # (left, right)
        self.label = label
        self.possible_labels = possible_labels
        self.possible_labels.add(label)


class Lower(Node):
    def __init__(self, label: int, upper: Node):
        super().__init__(label)
        self.upper = upper


def D(p: Pixel, q: Pixel):
    # TODO help needed
    # p -> pixel from left img (steady); q -> p's image from right img (movable)
    return np.mean(np.absolute(p.intensity[0] - q.intensity[1]))


def V(p: Pixel, q: Pixel, K=80, sigma=5):
    return K*max(sigma, abs(p.label - q.label))

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
