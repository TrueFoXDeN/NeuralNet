import networkx as nx

from multilayer_perceptron import NeuralNetwork
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def plot_neural_network(nn: NeuralNetwork, activations: List[np.ndarray]):
    G = nx.DiGraph()

    layer_sizes = [nn.input_size] + nn.hidden_layers + [nn.output_size]
    pos = {}
    node_colors = []

    # Create nodes and edges of the network
    for layer_idx, layer_size in enumerate(layer_sizes):
        for neuron_idx in range(layer_size):
            node_id = (layer_idx, neuron_idx)
            G.add_node(node_id)
            pos[node_id] = (layer_idx, -neuron_idx)
            if layer_idx < len(activations):
                activation = activations[layer_idx][neuron_idx]
                node_colors.append(activation)

    # Create edges between layers
    for layer_idx in range(len(layer_sizes) - 1):
        for src_idx in range(layer_sizes[layer_idx]):
            for dst_idx in range(layer_sizes[layer_idx + 1]):
                src_node = (layer_idx, src_idx)
                dst_node = (layer_idx + 1, dst_idx)
                G.add_edge(src_node, dst_node)

    # Draw the network
    plt.figure(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis, node_size=100)
    edges = nx.draw_networkx_edges(G, pos)
    plt.colorbar(nodes)
    plt.title('Neural Network')
    plt.show()