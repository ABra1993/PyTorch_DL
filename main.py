from network_modules.MNIST import MNIST_dataset
from network_modules import MNIST_network
import torch


def main():

    # Define params
    batch_size_train = 64
    batch_size_test = 128
    train_epochs = 3
    learning_rate = 1e-3
    momentum = 0.9
    visualize_training_data = False

    # Imports MNIST dataset
    MNIST = MNIST_dataset(batch_size_train=batch_size_train,
                          batch_size_test=batch_size_test,
                          train_epochs=train_epochs)
    MNIST.import_data()

    # Visualize training data
    if visualize_training_data == True:
        MNIST.visualize_training_data()

    # Initializes network
    network = MNIST_network.Net()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # Trains network
    network = MNIST.train(network, optimizer)

    # Evaluates network
    MNIST.test(network)

    # Plots result of training and test data
    MNIST.result()


if __name__ == "__main__":
    main()
