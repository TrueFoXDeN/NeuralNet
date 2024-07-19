import numpy as np
from sklearn.datasets import load_digits
from multilayer_perceptron import NeuralNetwork, normalize
from plot import plot_neural_network

if __name__ == '__main__':
    nn = NeuralNetwork(8*8, [16, 16], 10)

    digits = load_digits()
    digits = digits.data[0] / 15

    # initial_activation = normalize(np.random.rand(1, 8)[0])
    activations = nn.feedforward(np.array(digits))
    print(activations[-1])
    plot_neural_network(nn, activations)



    # plt.gray()
    # plt.matshow(digits.images[0])
    # plt.show()
