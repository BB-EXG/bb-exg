# One-Shot Compression of Large Edge-Exchangeable Graphs using Bits-Back Coding.

We present an optimal one-shot method for compressing large network graphs called Bits-Back for Edge-Exchangeable Graphs. The worst-case computational and memory complexities of our method scale quasi-linearly and linearly with the number of observed edges in the graph, making it efficient for compressing sparse real-world networks. Key to our method is bits-back coding, which is used to sample edges and vertices without replacement from the graph's edge-list in a way that preserves the structure of the network. Optimality is proven under a class of edge-exchangeable random graph models and experiments show the achieved bits-per-edge reaches the information content. We achieve competitive compression performance on real-world network datasets with millions of nodes and edges. The model used has only one parameter and can be learned via maximum likelihood with a simple line search.

# How to run the experiments in the paper.

## 1. Define the data directory and download datasets
```bash
export DATA_DIR=your_data_dir
./download_datasets.sh
```

## 2. Install dependencies
```bash
./install_dependencies.sh
```

## 3. Run experiments by specifying the datasets
```bash
python -B experiments/encode-decode-graph/run_experiment.py \
     --config=experiments/encode-decode-graph/configs/config_fhwm.py \
     --config.dataset_name=youtube
```

The complete list of datasets is available at [bb-exg/datasets.py](https://github.com/BB-EXG/bb-exg/blob/dev/bbexg/datasets.py#L11-L21)
