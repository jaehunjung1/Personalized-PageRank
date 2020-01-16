from absl import app
import dgl
import dgl.function as fn
import networkx as nx
import torch
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

    N = 100  # number of nodes
    N_2 = 10  # number of the second graph's nodes
    damp = 0.5  # damping factor
    K = 10  # number of power iterations

    # graph = dgl.DGLGraph()
    # graph.add_nodes(100)
    # graph.add_edges([i for i in range(6, 100)] + [1, 5, 3, 4, 5, 1], [4] * 94 + [0, 0, 2, 2, 2, 3])
    # graph.add_edges([4] * 94 + [0, 0, 2, 2, 2, 1], [i for i in range(6, 100)] + [1, 5, 3, 4, 5, 3])
    # graph.ndata['pv'] = (torch.ones(N, 1) / N).to(device)
    # graph.ndata['deg'] = graph.out_degrees(graph.nodes()).float().view(N, 1).to(device)

    graph = dgl.DGLGraph(nx.nx.star_graph(N_2 - 1))
    graph.ndata['pv'] = (torch.ones(N_2, 1) / N_2).to(device)
    graph.ndata['deg'] = graph.out_degrees(graph.nodes()).float().view(N_2, 1).to(device)
    graph_2 = dgl.DGLGraph(nx.nx.connected_watts_strogatz_graph(N_2, k=3, p=0.1, seed=999))
    graph_2.ndata['pv'] = (torch.ones(N_2, 1) / N_2).to(device)
    graph_2.ndata['deg'] = graph_2.out_degrees(graph_2.nodes()).float().view(N_2, 1).to(device)

    batch_graph = dgl.batch([graph, graph_2])

    print("Graph prepared.")

    personalization = torch.zeros(batch_graph.number_of_nodes(), 1)
    personalization[0] = 0.5
    personalization[1] = 0.5

    # the below two are first and second node for graph_2
    personalization[10] = 0.5
    personalization[11] = 0.5

    for i in range(K):
        pagerank_helper(device, batch_graph, personalization=personalization, damp=damp)

    print(batch_graph.ndata['pv'])
    # Todo unbatch the batch, and retrieve pv tensors from them.
    #      or just leave them as batch, retrieve pv tensor as concatenated, then slice them according to sizes

    draw_dgl_graph(batch_graph)


if __name__ == "__main__":
    app.run(main)
