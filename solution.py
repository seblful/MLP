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
            weight_size = (
                self.layers_sizes[num_neurons_ind + 1], self.layers_sizes[num_neurons_ind])
            bias_size = (self.layers_sizes[num_neurons_ind + 1], 1)

            random_weight = np.random.uniform(size=weight_size)
            random_bias = np.random.uniform(size=bias_size)

            self.weights.append(random_weight)
            self.biases.append(random_bias)

    def forward_propagation(self, x):
        a = x
        activations = [a]
        zs = []

        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, a) + bias
            a = self.sigmoid(z)

            zs.append(z)
            activations.append(a)

        return activations, zs

    def backward_propagation(self, x, y):
        activations, zs = self.forward_propagation(x)
        delta_weights = [np.zeros(weight.shape) for weight in self.weights]
        delta_biases = [np.zeros(bias.shape) for bias in self.biases]

        delta = self.cost_derivative(
            activations[-1], y) * self.sigmoid_derivative(zs[-1])
        delta_weights[-1] = np.dot(delta, activations[-2].T)
        delta_biases[-1] = delta

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            delta_weights[-l] = np.dot(delta, activations[-l - 1].T)
            delta_biases[-l] = delta

        return delta_weights, delta_biases

    def update_weights_and_biases(self, delta_weights, delta_biases):
        self.weights = [weight - (self.lr * delta_weight)
                        for weight, delta_weight in zip(self.weights, delta_weights)]
        self.biases = [bias - (self.lr * delta_bias)
                       for bias, delta_bias in zip(self.biases, delta_biases)]

    def train(self):
        for epoch in range(self.num_epochs):
            for x_batch, y_batch in zip(self.x_batches, self.y_batches):
                delta_weights_sum = [np.zeros(weight.shape)
                                     for weight in self.weights]
                delta_biases_sum = [np.zeros(bias.shape)
                                    for bias in self.biases]

                for x, y in zip(x_batch, y_batch):
                    delta_weights, delta_biases = self.backward_propagation(
                        x, y)
                    delta_weights_sum = [
                        dw + dw_batch for dw, dw_batch in zip(delta_weights_sum, delta_weights)]
                    delta_biases_sum = [
                        db + db_batch for db, db_batch in zip(delta_biases_sum, delta_biases)]

                delta_weights_avg = [
                    dw / self.batch_size for dw in delta_weights_sum]
                delta_biases_avg = [
                    db / self.batch_size for db in delta_biases_sum]

                self.update_weights_and_biases(
                    delta_weights_avg, delta_biases_avg)

    def predict(self, x):
        activations, _ = self.forward_propagation(x)
        return np.argmax(activations[-1])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost_derivative(self, output_activations, y):
        return output_activations - y


mlp = MLP()
mlp.train()

# Example usage:
x_test = np.random.uniform(size=(784, 1))
