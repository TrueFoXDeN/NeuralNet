from typing import List

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    def feedforward(self, activation: np.ndarray):
        for weights, biases in zip(self.weights, self.biases):
            activation = sigmoid(np.dot(activation, weights) + biases)
        return activation


nn = NeuralNetwork(3, [4, 5], 2)

print(nn.feedforward(np.array([1, 3, 2])))
# print(nn.weights)
# digits = load_digits()
# print(digits.data[0])

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()
