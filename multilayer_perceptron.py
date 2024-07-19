from typing import List

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


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

        # Iteriert über alle
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.rand(hidden_layers[i-1], hidden_layers[i]))
            self.biases.append(np.random.rand(hidden_layers[i]))

        self.weights.append(np.random.rand(hidden_layers[-1], output_size))
        self.biases.append(np.random.rand(output_size))


nn = NeuralNetwork(5, [2,  3], 4)
print(nn.weights)
# digits = load_digits()
# print(digits.data[0])

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()
