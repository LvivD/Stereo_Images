import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import numpy


class Graph:
    def __init__(self, edges=[]):
        self.graph = nx.Graph()
        if edges:
            self.createGraph(edges)

    def createGraph(self, edges):
        self.addEdge(edges)

    def addEdge(self, edges: tuple or list):
        for edge in edges:
            self.graph.add_edge(edge[0], edge[1], capacity=edge[2])

    # def cut(self, terminalA: str, terminalB: str, flow_function=edmonds_karp) -> tuple:
    #     cut_value, partition = nx.minimum_cut(
    #         self.graph, terminalA, terminalB, flow_func=edmonds_karp)
    #     reachable, non_reachable = partition

    #     cutset = set()
    #     for u, nbrs in ((n, self.graph[n]) for n in reachable):
    #         cutset.update((u, v) for v in nbrs if v in non_reachable)

    #     return cut_value, cutset

    def cut(self, weight="capacity"):
        return nx.stoer_wagner(self.graph, weight=weight)


class Pixel:
    def __init__(self, label, intensity):
        self.i = intensity
        self.l = label


def D(pixel, label):
    # TODO help needed
    return (pixel-label)**2


def V(p: Pixel, q: Pixel, K=7, sigma=5):
    if p.l != q.l:
        return (2*K) if (abs(q.i-p.i) <= sigma) else K
    return 0


# def treminalWeight(p: Pixel, labeling_ab, neighbours):
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
    print(G.cut('x', 'y'))
