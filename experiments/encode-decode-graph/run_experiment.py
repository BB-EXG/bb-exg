from ml_collections import config_flags
from ml_collections import config_dict
from bbexg.models import BBEXGCodec, FiniteHollywoodModelCodec, compute_fhm_info_content
from bbexg.datasets import load_dataset
from absl import app
from absl import flags

import craystack as cs
import scipy
import random

_CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_list("tags", None, "Tags for the experiment. (comma separated)")
flags.DEFINE_bool("decode", False, "Decode the graph.")
_FLAGS = flags.FLAGS


def main(_):
    print(f"Running experiment with config: \n{_CONFIG.value}")

    # You can log the results to your favorite experiment tracking tool here.
    results = run_experiment(_CONFIG.value)


def run_experiment(config: config_dict.ConfigDict):

    # Load network data.
    graph = load_dataset(
        dataset_name=config.dataset_name, max_num_edges=config.max_num_edges
    )

    # Randomly permute edges in the graph.
    # This is necessary to ensure the binary search tree used to compute the CDFs is reasonably balanced.
    random.seed(config.seed)
    random.shuffle(graph.edge_list)

    # Optimize bias for the given graph.
    def objective(bias):
        _, graph_info_content = compute_fhm_info_content(bias, graph)
        bpe = graph_info_content / graph.num_edges
        print(f'bpe: {bpe}, bias: {bias}')
        return bpe

    print(f"Optimizing bias for {config.dataset_name}")
    results = scipy.optimize.minimize_scalar(
        objective, bounds=(0, 10), method="bounded"
    )
    optimal_bias = results.x
    print(f"Optimal bias: {optimal_bias}")

    # Initialize the vertex and graph codecs.
    vertex_codec = FiniteHollywoodModelCodec(num_nodes=graph.num_nodes, bias=optimal_bias)
    graph_model = BBEXGCodec(
        num_nodes=graph.num_nodes,
        num_edges=graph.num_edges,
        vertex_codec=vertex_codec,
        name=config.dataset_name,
    )

    # Encode the graph using ANS.
    # Note that we use a single state for the entire graph.
    # The number of edges (`m` in the paper) and the bias are accounted for by adding 32 bits for each.
    ans_state = cs.rans.base_message(shape=(1,))
    ans_state = graph_model.push(ans_state, graph)
    bpe = (64 + 32 * len(cs.flatten(ans_state))) / graph.num_edges
    seq_info_content, graph_info_content = compute_fhm_info_content(optimal_bias, graph)

    print(f"{config.dataset_name} bits per edge (actual): {bpe}")
    print(f"{config.dataset_name} bits per edge (info content graph): {graph_info_content/graph.num_edges}")
    print(f"{config.dataset_name} bits per edge (info content seq. ): {seq_info_content/graph.num_edges}")

    if _FLAGS.decode:
        # Decode the graph using ANS and check if it matches the original graph.
        _, graph_decoded = graph_model.pop(ans_state)
        decoding_correct = graph_decoded == graph
        print(f"{config.dataset_name} decoded graph == original graph: {decoding_correct}")
    else:
        print('Decoding skipped. (Use --decode to decode graph.)')
        decoding_correct = None

    return {
        "bpe": bpe,
        "graph_info_content": graph_info_content / graph.num_edges,
        "seq_info_content": seq_info_content / graph.num_edges,
        "decoding_correct": decoding_correct,
        "optimal_bias": optimal_bias,
    }


if __name__ == "__main__":
    app.run(main)
