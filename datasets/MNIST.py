import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import *
from datasets import Net

class MNIST_dataset(object):
    """ Builds and fits model for MNIST dataset"""

    def __init__(self, batch_size_train, batch_size_test, momentum=0.9):
        self.train_data = []
        self.test_data = []

        self.epoch = 10
        self.learning_rate = 0.01
        self.momentum = momentum

        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

        self.train_losses = []
        self.train_counter = []

    def import_data(self):
        """ Import test and training data"""

        # import training dataset
        self.train_data = DataLoader(torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
                                                                transform=transforms.ToTensor()),
                                     batch_size=self.batch_size_train, shuffle=False)

        #  import test dataset
        self.test_data = DataLoader(torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
                                                                transform=transforms.ToTensor()),
                                     batch_size=self.batch_size_test, shuffle=False)

    def train(self):
        """ Training model"""

        network = Net.Net()
        optimizer = torch.optim.SGD(network.parameters(), lr=self.learning_rate)

        # train model
        network.train()
        for batch_idx, (data, target) in enumerate(self.train_data):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            self.train_losses.append(loss.item())

    def visualize_training_data(self):
        """ Visualize training data"""

        examples = enumerate(self.train_data)
        batch_idx, (example_data, example_targets) = next(examples)

        plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray')
            plt.title("Ground truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def result(self):
        plt.figure()
        plt.plot(self.train_losses)
        plt.show()
