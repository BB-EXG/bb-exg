import fastremap
import numpy as np
from tqdm import tqdm
from typing import Optional
from .definitions import Graph, Codec
from joblib import Memory
import os

DATA_DIR =  os.environ['DATA_DIR']

AVAILABLE_DATASETS = [
    # social networks
    'youtube',
    'foursquare',
    'digg',
    'gowalla',

    # others
    'skitter',
    'dblp'
]

def load_dataset(dataset_name: str, max_num_edges=None) -> Graph:
    bbexg_data_dir = f'{DATA_DIR}/bbexg'

    if dataset_name == 'youtube':
        return load_youtube(f'{bbexg_data_dir}/youtube-u-growth/out.youtube-u-growth', max_num_edges)
    elif dataset_name == 'foursquare':
        return load_network_repo_dataset(f'{bbexg_data_dir}/soc-FourSquare.mtx', max_num_edges)
    elif dataset_name == 'digg':
        return load_network_repo_dataset(f'{bbexg_data_dir}/soc-digg.mtx', max_num_edges)
    elif dataset_name == 'gowalla':
        return load_SNAP_dataset(f'{bbexg_data_dir}/loc-gowalla_edges.txt', max_num_edges)
    elif dataset_name == 'skitter':
        return load_SNAP_dataset(f'{bbexg_data_dir}/as-skitter.txt', max_num_edges)
    elif dataset_name == 'dblp':
        return load_SNAP_dataset(f'{bbexg_data_dir}/com-dblp.ungraph.txt', max_num_edges)
    else:
        raise ValueError(f'Dataset {dataset_name} is not available.')

def relabel_vertices(edge_list: list) -> list:
    vertex_array, _ = fastremap.renumber(np.array(edge_list, dtype=np.uint32) - 1)
    edge_list = vertex_array.tolist()
    return edge_list


def load_network_repo_dataset(path: str, max_num_edges: Optional[int] = None):
    with open(path, 'r') as f:
        txt = f.readlines()[2:]

    edge_list = list()
    for i, line in tqdm(enumerate(txt, start=1)):
        v, w = line.replace('\n', '').split(' ')[:2]
        edge_list.append((int(v), int(w)))

        if i == max_num_edges:
            break

    edge_list = relabel_vertices(edge_list)
    num_nodes = max(map(max, edge_list)) + 1

    return Graph(
        edge_list=edge_list,
        num_nodes=num_nodes,
        num_edges=len(edge_list)
    )


def load_youtube(path: str, max_num_edges: Optional[int] = None):
    with open(path, 'r') as f:
        txt = f.readlines()

    edge_list = list()
    i = 0
    for line in tqdm(txt):
        if line.startswith('%'):
            continue

        i += 1
        v, w = line.replace('\n', '').split(' ')[:2]
        edge_list.append((int(v), int(w)))

        if i == max_num_edges:
            break

    edge_list = relabel_vertices(edge_list)
    num_nodes = max(map(max, edge_list)) + 1

    return Graph(
        edge_list=edge_list,
        num_nodes=num_nodes,
        num_edges=len(edge_list)
    )


def load_SNAP_dataset(path: str, max_num_edges: Optional[int] = None) -> Graph:
    with open(path, 'r') as f:
         txt = f.readlines()
    
    # parse file to get vertex pairs
    edge_list = list()
    i = 0
    for line in tqdm(txt):
        if line.startswith('#'):
            continue

        i += 1
        v, w = line.replace('\n', '').split('\t')
        edge_list.append((int(v), int(w)))

        if i == max_num_edges:
            break

    # relabel vertices to be 0, 1, 2, ...
    edge_list = relabel_vertices(edge_list)
    num_nodes = max(map(max, edge_list)) + 1

    return Graph(
        edge_list=edge_list,
        num_nodes=num_nodes,
        num_edges=len(edge_list)
    )
