from network_modules.MNIST import MNIST_dataset
from network_modules import MNIST_network
import torch


def main():

    # Define params
    batch_size_train = 128
    batch_size_test = 10000
    train_epochs = 5
    learning_rate = 1e-2
    momentum = 0

    # additional information
    visualize_training_data = False
    show_parameters = False
    train_network = True

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
    optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate, momentum=momentum)

    # Prints weights and biased
    if show_parameters == True:
        MNIST.show_parameters(network)

    # Trains network
    if train_network == True:
        network = MNIST.train(network, optimizer)

    # Evaluates network
    MNIST.test(network)

    # Plots result of training and test data
    MNIST.result()


if __name__ == "__main__":
    main()
