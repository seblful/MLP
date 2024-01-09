import numpy as np


class MLP:
    def __init__(self,
                 train_data,
                 train_labels,
                 val_data,
                 val_labels,
                 layers=[3, 5, 2],
                 batch_size=10,
                 lr=0.01,
                 num_epochs=100):

        # self.train_data = train_data.reshape(
        #     train_data.shape[0], train_data.shape[1], 1)
        self.train_data = train_data
        self.train_labels = train_labels

        self.val_data = val_data
        self.val_labels = val_labels

        self.layers_sizes = [self.train_data.shape[1]] + layers
        self.num_layers = len(self.layers_sizes)

        self.batch_size = 10
        self.number_of_chunks = len(self.train_data) // self.batch_size

        self.x_batches = np.split(
            self.train_data, self.number_of_chunks, axis=0)
        self.y_batches = np.split(
            self.train_labels, self.number_of_chunks, axis=0)

        self.lr = 0.01
        self.num_epochs = 100

        self.weights = []
        self.biases = []

        self.init_weights_and_biases()

        print(self.train_data.shape)
        print([arr.shape for arr in self.weights])
        print([arr.shape for arr in self.biases])
        print("*" * 10, "init done")

    def init_weights_and_biases(self):
        for num_neurons_ind in range(len(self.layers_sizes) - 1):

            weight_size = (self.layers_sizes[num_neurons_ind + 1],
                           self.layers_sizes[num_neurons_ind])
            bias_size = (self.layers_sizes[num_neurons_ind + 1], 1)

            random_weight = np.random.uniform(size=weight_size)
            random_bias = np.random.uniform(size=bias_size)

            self.weights.append(random_weight)
            self.biases.append(random_bias)

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_deriv(self, x):
        return x > 0

    def softmax(self, x):
        a = np.exp(x) / sum(np.exp(x))
        return a

    def forward_propagation(self, x):
        activations = [x]

        for layer in range(self.num_layers - 2):
            print(self.weights[layer].shape, activations[layer].shape)
            z = np.dot(self.weights[layer],
                       activations[layer]) + self.biases[layer]
            a = self.relu(z)
            activations.append(a)

        print(self.weights[-1].shape, activations[-1].shape)
        output = self.softmax(
            np.dot(self.weights[-1], activations[-1]) + self.biases[-1])
        return activations, output

    def backward_propagation(self, activations, output, y):
        m = y.shape[0]
        grads_weights = []
        grads_biases = []
        print(output)
        delta = output - y
        grads_weights.append(np.dot(delta, activations[-2].T) / m)
        grads_biases.append(np.mean(delta, axis=1, keepdims=True))
        for l in range(self.num_layers - 3, -1, -1):
            delta = np.dot(self.weights[l + 1].T, delta) * \
                self.relu_deriv(activations[l + 1])
            grads_weights.insert(0, np.dot(delta, activations[l].T) / m)
            grads_biases.insert(0, np.mean(delta, axis=1, keepdims=True))
        return grads_weights, grads_biases

    def update_parameters(self, grads_weights, grads_biases):
        for layer in range(self.num_layers - 1):
            self.weights[layer] -= self.lr * grads_weights[layer]
            self.biases[layer] -= self.lr * grads_biases[layer]

    def train(self):
        for epoch in range(self.num_epochs):
            if epoch % 10 == 0:
                print(
                    f"Training on epoch {epoch + 1} from {self.num_epochs} epochs...")
            for batch in range(self.number_of_chunks):
                x_batch = self.x_batches[batch]
                y_batch = self.y_batches[batch]
                activations, output = self.forward_propagation(x_batch.T)
                grads_weights, grads_biases = self.backward_propagation(
                    activations, output.T, y_batch.T)
                self.update_parameters(grads_weights, grads_biases)

    def predict(self, x):
        _, output = self.forward_propagation(x.T)
        return np.argmax(output, axis=0)
