import numpy as np


class MLP:
    def __init__(self,
                 train_data,
                 train_labels,
                 val_data,
                 val_labels,
                 num_classes=10,
                 batch_size=3200,
                 lr=0.001,
                 num_epochs=100):

        self.train_data = train_data
        self.train_labels = train_labels

        self.val_data = val_data
        self.val_labels = val_labels

        self.num_classes = num_classes

        self.layers_sizes = [10, num_classes]
        self.num_layers = len(self.layers_sizes)

        self.batch_size = batch_size
        self.num_batches = len(train_data) // self.batch_size
        self.batch_indices = np.random.permutation(len(train_data))

        self.lr = lr
        self.num_epochs = num_epochs

        self.init_weights_and_biases()

        # print(self.train_data.shape, self.W1.shape,
        #       self.b1.shape, self.W2.shape, self.b2.shape)

    def init_weights_and_biases(self):
        self.W1 = np.random.random(
            size=(self.layers_sizes[0], self.train_data.shape[1]))
        self.b1 = np.random.random(size=(self.layers_sizes[0], 1))
        self.W2 = np.random.random(
            size=(self.layers_sizes[1], self.layers_sizes[0]))
        self.b2 = np.random.random(size=(self.layers_sizes[1], 1))

    def relu(self, z):
        return np.maximum(z, 0)

    def relu_deriv(self, x):
        return x > 0

    def softmax(self, z):
        a = np.exp(z) / sum(np.exp(z))
        print(a)
        return a

    def one_hot(self, y):
        one_hot_y = np.zeros((y.size, y.max() + 1))
        one_hot_y[np.arange(y.size), y] = 1
        one_hot_y = one_hot_y.T
        return one_hot_y

    def forward_propagation(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2

    def backward_propagation(self, Z1, A1, Z2, A2, X, y):
        one_hot_y = self.one_hot(y)

        dZ2 = A2 - one_hot_y
        dW2 = dZ2 @ A1.T / self.num_classes
        db2 = np.sum(dZ2, axis=1, keepdims=True) / self.num_classes

        dZ1 = self.W2.T @ dZ2 * self.relu_deriv(Z1)
        dW1 = dZ1 @ X.T / self.num_classes
        db1 = np.sum(dZ1, axis=1, keepdims=True) / self.num_classes

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, a):
        return np.argmax(a, 0)

    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def train(self):
        for epoch in range(self.num_epochs):
            # Iterate over batches
            for batch_idx in range(self.num_batches):
                # Get the indices of the current batch
                batch_indices = self.batch_indices[batch_idx *
                                                   self.batch_size: (batch_idx + 1) * self.batch_size]

                # Extract batch data
                x_batch = self.train_data[batch_indices]
                y_batch = self.train_labels[batch_indices]

                Z1, A1, Z2, A2 = self.forward_propagation(x_batch.T)
                dW1, db1, dW2, db2 = self.backward_propagation(
                    Z1, A1, Z2, A2, x_batch.T, y_batch)
                self.update_parameters(dW1, db1, dW2, db2)

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch + 1} from {self.num_epochs} epochs...")

                    predictions = self.predict(A2)
                    print(self.get_accuracy(predictions, y_batch))
