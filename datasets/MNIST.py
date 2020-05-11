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

    def __init__(self, batch_size_train, batch_size_test):
        self.train_data = []
        self.test_data = []

        self.epoch = 3
        self.log_interval = 3

        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = []

    def import_data(self):
        """ Import test and training data"""

        # import training dataset
        self.train_data = DataLoader(torchvision.datasets.MNIST('MNIST/processed/training.pt', train=True, download=True,
                                                                transform=transforms.ToTensor()),
                                     batch_size=self.batch_size_train, shuffle=False)

        #  import test dataset
        self.test_data = DataLoader(torchvision.datasets.MNIST('MNIST/processed/test.pt', train=False, download=True,
                                                                transform=transforms.ToTensor()),
                                     batch_size=self.batch_size_test, shuffle=False)

        self.test_counter = [i*len(self.train_data.dataset) for i in range(self.epoch + 1)]

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

        print(example_data[i])

    def train(self, network, optimizer):
        """ Training model"""

        network.train()

        for i in range(0, self.epoch):
            for batch_idx, (data, target) in enumerate(self.train_data):
                optimizer.zero_grad()
                output = network(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} of {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                        i + 1, self.epoch, batch_idx * len(data), len(self.train_data.dataset),
                        100. * batch_idx / len(self.train_data), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                    (batch_idx*self.batch_size_train) + ((self.epoch-1)*len(self.train_data.dataset)))

        return network

    def test(self, network):
        """ Tests network on the test set"""

        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_data:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_data.dataset)
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(self.test_data.dataset),
            100. * correct / len(self.test_data.dataset)))

    def result(self):
        """ Plots the loss of training and test data"""

        plt.figure()
        plt.grid()
        plt.plot(self.train_losses, 'k')
        plt.title('Loss')
        plt.xlabel('Batch ID')
        plt.ylabel('Negative log likelihood loss')
        plt.show()


