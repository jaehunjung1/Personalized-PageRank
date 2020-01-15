import dgl
import networkx as nx
import matplotlib.pyplot as plt
import ipdb


def draw_dgl_graph(graph: dgl.DGLGraph) -> None:
    nx_graph = graph.to_networkx()

    nx.draw(nx_graph, with_labels=True, node_size=50, node_color=[[.9, .9, .9]])
    plt.show(block=True)
