import networkx as nx

from multilayer_perceptron import NeuralNetwork
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_neural_network(nn: NeuralNetwork, activations: List[np.ndarray]):
    G = nx.DiGraph()

    layer_sizes = [nn.input_size] + nn.hidden_layers + [nn.output_size]
    pos = {}
    node_colors = []

    # Create custom colormap from red to blue
    cmap = LinearSegmentedColormap.from_list('red_green', ['red', 'orange', 'green'], N=256)

    # Create nodes and edges of the network
    for layer_idx, layer_size in enumerate(layer_sizes):
        layer_height = layer_size
        y_offset = layer_height / 2.0 - 0.5
        for neuron_idx in range(layer_size):
            node_id = (layer_idx, neuron_idx)
            G.add_node(node_id)
            pos[node_id] = (layer_idx, y_offset - neuron_idx)
            if layer_idx < len(activations):
                activation = activations[layer_idx][neuron_idx]
                node_colors.append(activation)

    # Normalize node colors
    node_colors = np.array(node_colors)
    node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())

    # Create edges between layers
    for layer_idx in range(len(layer_sizes) - 1):
        weights = nn.weights[layer_idx]
        for src_idx in range(layer_sizes[layer_idx]):
            for dst_idx in range(layer_sizes[layer_idx + 1]):
                src_node = (layer_idx, src_idx)
                dst_node = (layer_idx + 1, dst_idx)
                G.add_edge(src_node, dst_node, weight=weights[src_idx, dst_idx])

    # Draw the network
    plt.figure(figsize=(24, 32))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, node_size=200)

    # Extract weights and normalize for alpha values
    edges = []
    alphas = []
    for (u, v, d) in G.edges(data=True):
        edges.append((u, v))
        alphas.append(abs(d['weight']))

    # Normalize alpha values
    alphas = np.array(alphas)
    if alphas.max() > alphas.min():
        alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())
    else:
        alphas.fill(1.0)  # If all weights are the same, make all alphas 1.0

    for i, (edge, alpha) in enumerate(zip(edges, alphas)):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], alpha=alpha, edge_color='k')

    plt.title('Neural Network')
    plt.show()