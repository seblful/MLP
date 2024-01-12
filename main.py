from MLP_numpy import MLP
import mnist


def create_mnist():
    # mnist.init()
    x_train, y_train, x_test, y_test = mnist.load()

    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


def main():
    # Create data
    x_train, y_train, x_test, y_test = create_mnist()
    mlp = MLP(train_data=x_train,
              train_labels=y_train,
              val_data=x_test,
              val_labels=y_test,
              batch_size=3200,
              lr=0.001,
              num_epochs=500)

    # Train multilayer perceptron
    mlp.train()

    # Visualize accuracy over epochs
    mlp.visualize_training()

    # Predict test image
    mlp.predict_image(x_test[0])


if __name__ == "__main__":
    main()
