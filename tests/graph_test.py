import networkx as nx
G = nx.DiGraph()
# G.add_edge('x','a', capacity = 3.0)
# G.add_edge('x','b', capacity = 1.0)
# G.add_edge('a','c', capacity = 3.0)
# G.add_edge('b','c', capacity = 5.0)
# G.add_edge('b','d', capacity = 4.0)
# G.add_edge('d','e', capacity = 2.0)
# G.add_edge('c','y', capacity = 2.0)
# G.add_edge('e','y', capacity = 3.0)
G.add_edge(11,12, capacity = float('inf'))
G.add_edge(21,22, capacity = float('inf'))
G.add_edge(31,32, capacity = float('inf'))
G.add_edge(41,42, capacity = float('inf'))
G.add_edge(12,21, capacity = 4)
G.add_edge(22,11, capacity = 4)
G.add_edge(12,31, capacity = 2)
G.add_edge(32,11, capacity = 2)
G.add_edge(22,31, capacity = 3)
G.add_edge(32,21, capacity = 3)
G.add_edge(22,41, capacity = 2)
G.add_edge(42,21, capacity = 2)
G.add_edge(32,41, capacity = 5)
G.add_edge(42,31, capacity = 5)

cut_value, partition = nx.minimum_cut(G, 11, 42)
print(partition)
print(cut_value)
reachable, non_reachable = partition
cutset = set()
for u, nbrs in ((n, G[n]) for n in reachable):    
    cutset.update((u, v) for v in nbrs if v in non_reachable)
print(sorted(cutset))

if G.has_edge('x', 'y'):
    G.remove_edge('x', 'y')
# print(cut_value == sum(G.edge[u][v]['capacity'] for (u, v) in cutset))





