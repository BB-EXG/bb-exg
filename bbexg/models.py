from .utils import (
    graph_to_vertex_list,
    multiset_from_graph,
    graph_from_multiset,
    flatten,
)
from .definitions import Codec, Graph
from tqdm import tqdm
from math import log2
from collections import defaultdict
from typing import Tuple
import random

import craystack as cs


def UniformCodec(prec: int) -> Codec:
    """Encodes/decodes a symbol using a uniform distribution.

    Args:
        prec: The precision of the ANS state. This should be set to the alphabet size of the uniform distribution.

    Returns:
        A uniform codec.
    """

    def push(ans_state, symbol):
        ans_state = cs.rans.push_with_finer_prec(ans_state, symbol, 1, prec)
        return ans_state

    def pop(ans_state, *context):
        symbol, pop_ = cs.rans.pop_with_finer_prec(ans_state, prec)
        ans_state = pop_(symbol, 1)
        return ans_state, int(symbol[0])

    return Codec(push, pop)


def BernoulliCodec(p, prec=16) -> Codec:
    """Encodes/decodes a symbol using a Bernoulli distribution with parameter `p`.

    Args:
        p: The parameter of the Bernoulli distribution.
        prec: The precision of the ANS state.

    Returns:
        A Bernoulli codec.

    """
    codec = cs.Bernoulli(p, prec=prec)

    def push(ans_state, symbol):
        (ans_state,) = codec.push(ans_state, int(symbol))
        return ans_state

    def pop(ans_state):
        ans_state, symbol = codec.pop(ans_state)
        return ans_state, int(symbol[0])

    return Codec(push, pop)


def compute_fhm_info_content(bias: float, graph: Graph) -> Tuple[float, float]:
    """Computes the sequence and graph information contents under the FHM model.

    Args:
        bias: The bias of the FHM model.
        graph: The graph to compute the information content of.

    Returns:
        The sequence and graph information contents.
    """
    vertex_counts = defaultdict(lambda: 0)
    sequence_info_content = 0.0
    num_unique_vertices = 0

    sequence_info_content = 0
    for step, vertex in enumerate(flatten(graph.edge_list)):
        vertex_count = vertex_counts[vertex]
        in_graph = vertex_count != 0

        # compute sequence info content for one vertex
        p = (step + num_unique_vertices * bias) / (graph.num_nodes * bias + step)
        sequence_info_content -= (
            0 if p in [0, 1] else in_graph * log2(p) + (1 - in_graph) * log2(1 - p)
        )
        if in_graph:
            sequence_info_content -= log2(vertex_count / step)
        else:
            sequence_info_content -= log2(1 / (graph.num_nodes - num_unique_vertices))

        # update vertex counts and num_unique_vertices
        vertex_counts[vertex] += 1
        if not in_graph:
            num_unique_vertices += 1

    num_bits_back = graph.num_edges + sum(
        log2(graph.num_edges - i) for i in range(graph.num_edges)
    )
    graph_info_content = sequence_info_content - num_bits_back
    return sequence_info_content, graph_info_content


def FiniteHollywoodModelCodec(num_nodes: int, bias: float) -> Codec:
    """Encodes/decodes a graph using the Finite Hollywood Model.

    Args:
        num_nodes: The number of nodes in the graph.
        bias: The bias of the FHM model.

    Returns:
        A FHM codec.

    """
    swor_codec = cs.multiset.SamplingWithoutReplacementCodec()
    frequency_codec = cs.multiset.FrequencyCountCodec()

    def bernoulli_codec(step, num_unique_vertices):
        return BernoulliCodec(
            (step + num_unique_vertices * bias) / (num_nodes * bias + step), prec=27
        )

    def push(ans_state, vertex, context):
        vertex_multiset, vertex_set = context

        # Update vertex multiset
        vertex_multiset = cs.multiset.remove(vertex_multiset, vertex)
        in_graph = cs.multiset.check_if_in_multiset(vertex, vertex_multiset)

        # Encode vertex
        if in_graph:
            ans_state = frequency_codec.push(ans_state, vertex, vertex_multiset)
        else:
            ans_state, vertex_set = swor_codec.push(ans_state, vertex, vertex_set)

        # Compute the number of unique nodes in the sequence.
        # The first element of `vertex_set` is equal to the total number of elements in `vertex_set`.
        num_unique_nodes = num_nodes - (vertex_set[0] if vertex_set else 0)

        vertex_multiset_size = vertex_multiset[0] if vertex_multiset else 0
        ans_state = bernoulli_codec(vertex_multiset_size, num_unique_nodes).push(
            ans_state, in_graph
        )

        context = (vertex_multiset, vertex_set)
        return ans_state, context

    def pop(ans_state, context):
        vertex_multiset, vertex_set = context
        vertex_multiset_size = vertex_multiset[0] if vertex_multiset else 0

        # Compute the number of unique nodes in the sequence.
        # The first element of `vertex_set` is equal to the total number of elements in `vertex_set`.
        num_unique_nodes = num_nodes - (vertex_set[0] if vertex_set else 0)

        # Decode vertex
        ans_state, in_graph = bernoulli_codec(
            vertex_multiset_size, num_unique_nodes
        ).pop(ans_state)
        if in_graph:
            ans_state, vertex = frequency_codec.pop(ans_state, vertex_multiset)
        else:
            ans_state, vertex, vertex_set = swor_codec.pop(ans_state, vertex_set)

        # Update vertex multiset
        vertex_multiset = cs.multiset.insert(vertex_multiset, vertex)
        context = (vertex_multiset, vertex_set)
        return ans_state, vertex, context

    return Codec(push, pop)


def BBEXGCodec(num_nodes: int, num_edges: int, vertex_codec: Codec, name=None) -> Codec:
    """Encodes/decodes a graph using BBEXG.

    Args:
        num_nodes: The number of nodes in the graph.
        num_edges: The number of edges in the graph.
        vertex_codec: The codec to use for encoding/decoding the vertices. See the FiniteHollywoodModelCodec function for an example.
        name: The name of the codec. Used only for logging purposes.

    Args:
        A BBEXG codec for encoding/decoding a graph using the given vertex codec.
    """
    binary_model = UniformCodec(2)
    swor_codec = cs.multiset.SamplingWithoutReplacementCodec()

    def push(ans_state, graph: Graph):
        # Build multisets
        graph_multiset = multiset_from_graph(graph)
        vertex_multiset = cs.multiset.build_multiset(graph_to_vertex_list(graph))
        vertex_set = ()
        context = (vertex_multiset, vertex_set)

        print(f"num_edges: {num_edges}")
        with tqdm(total=num_edges, desc=f"encoding {name}") as pbar:
            for step in range(graph.num_edges):
                # 1) Sample an edge without replacement
                ans_state, edge, graph_multiset = swor_codec.pop(
                    ans_state, graph_multiset
                )

                # 2) Pick an order for the vertices
                ans_state, i = binary_model.pop(ans_state)

                # 3) Encode vertex
                ans_state, context = vertex_codec.push(ans_state, edge[i], context)
                ans_state, context = vertex_codec.push(ans_state, edge[1 - i], context)

                if step % (num_edges // 20) == 0:
                    pbar.update((num_edges // 20))

        return ans_state

    def pop(ans_state):
        graph_multiset = ()
        vertex_multiset = ()

        # Initialize a set containing all nodes.
        # Shuffling is done to ensure that the binary search tree used to represent the set is balanced
        all_nodes = list(range(num_nodes))
        random.seed(0)
        random.shuffle(all_nodes)
        vertex_set = cs.multiset.build_multiset(all_nodes)

        context = (vertex_multiset, vertex_set)

        step = 0
        with tqdm(total=num_edges, desc=f"decoding {name}") as pbar:
            for step in range(num_edges):
                # Decode vertices
                ans_state, w, context = vertex_codec.pop(ans_state, context)
                ans_state, v, context = vertex_codec.pop(ans_state, context)

                # 2) Bits-back: infer order for the vertices
                if v < w:
                    i = 0
                    edge = [v, w]
                else:
                    i = 1
                    edge = [w, v]
                ans_state = binary_model.push(ans_state, i)

                # 1) Bits-back: input an edge without replacement
                ans_state, graph_multiset = swor_codec.push(
                    ans_state, edge, graph_multiset
                )

                if step % (num_edges // 20) == 0:
                    pbar.update((num_edges // 20))

        graph = graph_from_multiset(graph_multiset, num_nodes, num_edges)
        return ans_state, graph

    return Codec(push, pop)
