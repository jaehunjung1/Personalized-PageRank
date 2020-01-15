import dgl
import networkx as nx
import matplotlib.pyplot as plt


def draw_dgl_graph(graph: dgl.DGLGraph) -> None:
    nx.draw(graph.to_networkx(), node_size=50, node_color=[[.3, .3, .3]])
    plt.show()