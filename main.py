import dgl
import networkx as nx
import torch
import matplotlib.pyplot as plt
import ipdb

from util import *

N = 100  # number of nodes
DAMP = 0.85  # damping factor
K = 10  # number of power iterations

graph = dgl.DGLGraph(nx.nx.erdos_renyi_graph(N, 0.1))
graph.ndata['pv'] = torch.ones(N) / N
graph.ndata['deg'] = graph.out_degrees(graph.nodes()).float()
# draw_dgl_graph(graph)


def pagerank_message_func(edges):
    return {'pv': edges.src['pv'] / edges.src['deg']}


def pagerank_reduce_func(nodes):
    ipdb.set_trace()
    msgs = torch.sum(nodes.mailbox['pv'], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pv': pv}


def pagerank_batch(g):
    g.send(g.edges())
    g.recv(g.nodes())


graph.register_message_func(pagerank_message_func)
graph.register_reduce_func(pagerank_reduce_func)
pagerank_batch(graph)
