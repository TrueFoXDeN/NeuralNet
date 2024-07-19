from typing import List

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_layers: List, output_size: int):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.weights = []
        self.biases = []

        # Erstellt Matrix in der Größe von input_size x hidden_layers
        self.weights.append(np.random.rand(input_size, hidden_layers[0]))
        # Erstellt Bias Vektor in der Größe von input_size
        self.biases.append(np.random.rand(hidden_layers[0]))

        # Iteriert über alle hidden layer und fügt eine matrix an weights und einen vektor an biases hinzu
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.rand(hidden_layers[i - 1], hidden_layers[i]))
            self.biases.append(np.random.rand(hidden_layers[i]))

        self.weights.append(np.random.rand(hidden_layers[-1], output_size))
        self.biases.append(np.random.rand(output_size))

    def feedforward(self, activation: np.ndarray) -> List[np.ndarray]:
        activations = [activation]  # Initialize with input activation
        for weights, biases in zip(self.weights, self.biases):
            activation = sigmoid(np.dot(np.transpose(weights), activation) + biases)
            activations.append(activation)
        return activations


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
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis, node_size=500)
    edges = nx.draw_networkx_edges(G, pos)
    plt.colorbar(nodes)
    plt.title('Neural Network')
    plt.show()


nn = NeuralNetwork(5, [2, 3], 4)
initial_activation = normalize(np.array([1.9, 3.6, 2.2, 3.2, 0]))
activations = nn.feedforward(initial_activation)
print(activations[-1])
plot_neural_network(nn, activations)

# digits = load_digits()
# print(digits.data[0])

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()
