import numpy as np


class MLP:
    def __init__(self):
        self.input_shape = (28, 28)
        self.inputs = np.random.uniform(size=(8, 784, 1))
        self.outputs = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.resolution = int(self.input_shape[0] * self.input_shape[1])

        self.layers_sizes = [self.resolution, 2, 3, 5]
        self.num_layers = len(self.layers_sizes)

        self.weights = []
        self.biases = []

        self.init_weights_and_biases()

        self.lr = 0.01
        self.num_epochs = 1000

    def init_weights_and_biases(self):
        for num_neurons_ind in range(len(self.layers_sizes) - 1):

            weight_size = (self.layers_sizes[num_neurons_ind+1],
                           self.layers_sizes[num_neurons_ind])
            bias_size = (self.layers_sizes[num_neurons_ind+1], 1)

            random_weight = np.random.uniform(size=weight_size)
            random_bias = np.random.uniform(size=bias_size)

            self.weights.append(random_weight)
            self.biases.append(random_bias)

    def to_one_hot(self, outputs):
        diag_matr = np.eye(outputs.max() + 1)
        return diag_matr[outputs.reshape(-1)]

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_deriv(self, x):
        return x > 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        a = self.inputs
        for weight, bias in zip(self.weights, self.biases):
            # print(weight.shape, a.shape)
            z = weight @ a + bias
            a = self.relu(z)

        return a

    def backward(self, a):
        return
        dz = a

    def train(self):
        pass
