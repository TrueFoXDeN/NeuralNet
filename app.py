import numpy as np

from multilayer_perceptron import NeuralNetwork, normalize
from plot import plot_neural_network

if __name__ == '__main__':
    nn = NeuralNetwork(8, [4, 4], 10)
    initial_activation = normalize(np.random.rand(1, 8)[0])
    activations = nn.feedforward(initial_activation)
    print(activations[-1])
    plot_neural_network(nn, activations)
