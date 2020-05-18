# # import graph
# import networkx as nx
# # gr = graph.Graph([
# #     ('1', '2', 10),
# #     ('1', '3', 5),
# #     ('3', '2', 7),
# #     ('4', '2', 12),
# #     ('3', '4', 4),
# # ])

# # # res = gr.cut('1', '4')

# # res = gr.stoer()


# # print(res[0])

# # print(res[1])

# G = nx.Graph()
# G.add_edge('x', 'a', weight=3)
# G.add_edge('x', 'b', weight=1)
# G.add_edge('a', 'c', weight=3)
# G.add_edge('b', 'c', weight=5)
# G.add_edge('b', 'd', weight=4)
# G.add_edge('d', 'e', weight=2)
# G.add_edge('c', 'y', weight=2)
# G.add_edge('e', 'y', weight=3)
# cut_value, partition = nx.stoer_wagner(G)
# print(cut_value)


from graph import Graph

# gr = Graph([
#     ('1', '21', 10),
#     ('1', '31', 5),
#     ('21', '22', float('inf')),
#     ('31', '32', float('inf')),
#     ('22', '31', 7),
#     ('32', '21', 7),
#     ('22', '4', 12),
#     ('32', '4', 4),
# ])

gr = Graph([
    ('1', '2', 10),
    ('1', '3', 5),
    ('3', '2', 7),
    ('4', '2', 12),
    ('3', '4', 4),
])


res = gr.stoer()

print(res[0])

print(res[1])
