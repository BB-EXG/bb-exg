from dataclasses import dataclass
from typing import Tuple, Sequence
from collections import namedtuple

@dataclass
class Graph:
    edge_list: Sequence[Tuple[int, int]]
    num_nodes: int
    num_edges: int

    def __eq__(self, other):
        return sorted(map(sorted, self.edge_list)) == sorted(map(sorted, other.edge_list))

Codec = namedtuple('Model', ['push', 'pop'])