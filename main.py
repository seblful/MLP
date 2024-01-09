from MLP_numpy import MLP
import mnist


def create_mnist():
    # mnist.init()
    x_train, y_train, x_test, y_test = mnist.load()

    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


def main():

    x_train, y_train, x_test, y_test = create_mnist()
    mlp = MLP(train_data=x_train[:1000, :],
              train_labels=y_train[:1000],
              val_data=x_test[:500, :],
              val_labels=y_test[:500])

    mlp.train()

    prediction = mlp.predict(x_train[1002, :])
    print(prediction)


if __name__ == "__main__":
    main()
