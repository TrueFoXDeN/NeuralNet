from typing import List
import numpy as np


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

