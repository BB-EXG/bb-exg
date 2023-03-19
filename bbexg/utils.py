from .definitions import Graph
import craystack as cs


def flatten(l):
    return [item for sublist in l for item in sublist]


def graph_to_vertex_list(graph):
    return flatten(graph.edge_list)


def multiset_from_graph(graph: Graph):
    return cs.multiset.build_multiset(map(sorted, graph.edge_list))


def graph_from_multiset(multiset, num_nodes, num_edges):
    edge_list = cs.multiset.to_sequence(multiset)
    return Graph(
        edge_list=edge_list,
        num_nodes=num_nodes or max(map(max, multiset)) + 1,
        num_edges=num_edges or len(edge_list),
    )
