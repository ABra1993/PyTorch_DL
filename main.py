from datasets.MNIST import MNIST_dataset
from datasets import Net
import torch
import torchvision


def main():

    # Imports MNIST dataset
    MNIST = MNIST_dataset(batch_size_train=64, batch_size_test=128)
    MNIST.import_data()

    # Visualize training data
    #    MNIST.visualize_training_data()

    # Initializes network
    network = Net.Net()
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)

    # Trains network
    network = MNIST.train(network, optimizer)

    # Evaluates network
    MNIST.test(network)

    # Plots result of training and test data
    MNIST.result()


if __name__ == "__main__":
    main()
