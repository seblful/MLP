import numpy as np


class MLP:
    def __init__(self):
        self.input_shape = (28, 28)
        self.inputs = np.random.uniform(size=(12, 784, 1))
        self.outputs = np.array([0, 1, 1, 4, 6, 2, 5, 3, 4, 2, 0, 3])
        self.resolution = int(self.input_shape[0] * self.input_shape[1])

        self.layers_sizes = [self.resolution, 2, 3, 5]
        self.num_layers = len(self.layers_sizes)

        self.weights = []
        self.biases = []

        self.init_weights_and_biases()

        self.batch_size = 3
        self.number_of_chunks = len(self.inputs) // self.batch_size

        self.x_batches = np.split(self.inputs, self.number_of_chunks, axis=0)
        self.y_batches = np.split(self.outputs, self.number_of_chunks, axis=0)

        self.lr = 0.01
        self.num_epochs = 1000

    def init_weights_and_biases(self):
        for num_neurons_ind in range(len(self.layers_sizes) - 1):

            weight_size = (self.layers_sizes[num_neurons_ind + 1],
                           self.layers_sizes[num_neurons_ind])
            bias_size = (self.layers_sizes[num_neurons_ind + 1], 1)

            random_weight = np.random.uniform(size=weight_size)
            random_bias = np.random.uniform(size=bias_size)

            self.weights.append(random_weight)
            self.biases.append(random_bias)

    def to_one_hot(self, x):
        diag_matr = np.eye(x.max() + 1)
        return diag_matr[x.reshape(-1)]

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_deriv(self, x):
        return x > 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        a = np.exp(x) / sum(np.exp(x))
        return a

    def forward(self, batch):
        a = batch
        for weight, bias in zip(self.weights, self.biases):
            print(a.shape)
            # print(weight.shape, a.shape, bias.shape)
            z = weight @ a + bias
            a = self.relu(z)

        print(a.shape)

        a = self.sigmoid(a)

        print(a.shape)

        return a

    def backward(self, a, y_batch):
        # Compute the gradients of the weights and biases using backpropagation
        gradients = []

        # Compute the gradient of the last layer
        # print(a, self.to_one_hot(y_batch))
        delta = a - self.to_one_hot(y_batch)
        gradients.append(delta @ a.T)

        # Backpropagate the gradients through the layers
        for i in range(self.num_layers - 2, 0, -1):
            delta = (self.weights[i].T @ delta) * self.relu_deriv(a)
            gradients.append(delta @ a.T)

        # Reverse the gradients list to align with the weights list
        gradients.reverse()

        # Update the weights using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * gradients[i]

    def train(self):
        for epoch in range(self.num_epochs):
            for x_batch, y_batch in zip(self.x_batches, self.y_batches):
                # Forward pass
                a = self.forward(x_batch)

                # Backward pass
                self.backward(a, y_batch)
