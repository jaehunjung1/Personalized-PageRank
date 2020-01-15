import dgl
import dgl.function as fn
import networkx as nx
import torch
import matplotlib.pyplot as plt
import ipdb

from util import *

N = 100  # number of nodes
DAMP = 0.85  # damping factor
K = 3  # number of power iterations

graph = dgl.DGLGraph(nx.nx.erdos_renyi_graph(N, 0.1, seed=1234))
graph.ndata['pv'] = torch.ones(N, 1) / N
graph.ndata['deg'] = graph.out_degrees(graph.nodes()).float().view(N, 1)
# draw_dgl_graph(graph)


def pagerank_message_func(edges):
    return {'pv': edges.src['pv'] / edges.src['deg']}


def pagerank_reduce_func(nodes):
    msgs = torch.sum(nodes.mailbox['pv'], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pv': pv}


def pagerank_batch(graph: dgl.DGLGraph, personalization: torch.Tensor = None):
    if personalization:
        assert personalization.size(0) == graph.number_of_nodes(), "Personalization Vector should be sized (N, 1)"
    graph.ndata['pv'] = graph.ndata['pv'] / graph.ndata['deg']
    graph.update_all(message_func=fn.copy_src(src='pv', out='m'),
                     reduce_func=fn.sum(msg='m', out='m_sum'))
    graph.ndata['pv'] = (1 - DAMP) / N + DAMP * graph.ndata['m_sum']


for i in range(K):
    pagerank_batch(graph)

print(graph.ndata['pv'])
