from absl import app
import dgl
import dgl.function as fn
import networkx as nx
import torch
import matplotlib.pyplot as plt
import ipdb

from util import *


def pagerank_helper(device: torch.device,
                    graph: dgl.DGLGraph,
                    personalization: torch.Tensor = None,
                    damp: float = 0.8) -> torch.Tensor:
    """
    :param device: torch device
    :param graph: DGLGraph to compute PPR upon.
    :param personalization: Tensor sized (N, 1) which sums to 1, and represents element-wise bias to pagerank
    :param damp: damp factor for pagerank
    :return: updated page value tensor of the given graph
    """
    if personalization is not None:
        assert personalization.size(0) == graph.number_of_nodes(), "Personalization vector should be sized (N, 1)"
    else:
        N = graph.number_of_nodes()
        personalization = torch.ones(N, 1) / N
    personalization = personalization.to(device)

    graph.ndata['pv'] = graph.ndata['pv'] / graph.ndata['deg']
    graph.update_all(message_func=fn.copy_src(src='pv', out='m'),
                     reduce_func=fn.sum(msg='m', out='m_sum'))

    graph.ndata['pv'] = (1 - damp) * personalization + damp * graph.ndata['m_sum']

    return graph.ndata['pv']


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    N = 10  # number of nodes
    damp = 0.8  # damping factor
    K = 10  # number of power iterations

    graph = dgl.DGLGraph(nx.nx.erdos_renyi_graph(N, 0.1, seed=333))
    graph.ndata['pv'] = (torch.ones(N, 1) / N).to(device)
    graph.ndata['deg'] = graph.out_degrees(graph.nodes()).float().view(N, 1).to(device)
    # draw_dgl_graph(graph)

    print(graph.adjacency_matrix())

    print("graph prepared")

    personalization = torch.zeros(N, 1)
    personalization[0] = 0.5
    personalization[1] = 0.5

    for i in range(K):
        pagerank_helper(device, graph, personalization=personalization, damp=damp)

    print(graph.ndata['pv'])
    draw_dgl_graph(graph)


if __name__ == "__main__":
    app.run(main)
