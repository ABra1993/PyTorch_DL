import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import math


class MNIST_modules(object):
    """ Builds and fits model for MNIST dataset"""

    def __init__(self, batch_size_train, batch_size_test, train_epochs):
        self.train_data = []
        self.test_data = []

        self.train_epochs = train_epochs
        self.log_interval = 3

        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []

    def one_hot_encoding(self, output, target):
        """ Returns one-hot format for target"""

        target_one_hot = np.zeros((output.shape[0], output.shape[1]))

        for i in range(0, output.shape[0]):
            target_one_hot[i][target[i]] = 1

        return target_one_hot

    def import_data(self):
        """ Imports test and training data"""

        # import training data
        self.train_data = DataLoader(torchvision.datasets.MNIST('data/MNIST/processed/training.pt',
                                                                train=True,
                                                                download=True,
                                                                transform=transforms.ToTensor()),
                                                                batch_size=self.batch_size_train,
                                                                shuffle=False)

        #  import test data
        self.test_data = DataLoader(torchvision.datasets.MNIST('data/MNIST/processed/test.pt',
                                                               train=False,
                                                               download=True,
                                                               transform=transforms.ToTensor()),
                                                               batch_size=self.batch_size_test,
                                                               shuffle=False)

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

    def show_parameters(self, network):
        """ Prints weights and biases for every layer of the network"""

        for name, param in network.named_parameters():
            if param.requires_grad:
                print('\nParameter: {}\t ----\tSize: {}'.format(name, param.data.numpy().shape))

    def categorical_cross_entropy(self, output, target):
        """ Computes loss (objective function: categorical cross entropy)"""

        # Convert output and target tensors to numpy array
        output_array = output.detach().numpy()
        target_array = target.detach().numpy()

        # n = output_array.shape[0]
        # m = output_array.shape[1]

        n = self.batch_size_train
        m = 10

        # One-hot format for target
        target_one_hot = self.one_hot_encoding(output_array, target_array)

        for i in range(0, n):
            for j in range(0, m):
                output[i][j] = output[i][j].clone() * target_one_hot[i][j]

        loss = torch.mean(torch.sum(output, 1))

        return loss

    def train(self, network, optimizer, ce_torch):
        """ Training network on training data"""

        for i in range(0, self.train_epochs):
            for batch_idx, (data, target) in enumerate(self.train_data):
                optimizer.zero_grad()
                output = network(data)
                #print(output)
                if ce_torch == False:
                    loss = self.categorical_cross_entropy(output, target)
                else:
                    loss = F.cross_entropy(output, target)
                #print(loss)
                loss.backward()
                #sys.exit()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} of {} [{}/{} ({:.0f}%)]\t\tloss: {:.6f}'.format(
                        i + 1, self.train_epochs, batch_idx * len(data), len(self.train_data.dataset),
                        100. * batch_idx / len(self.train_data), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append(
                    (batch_idx*self.batch_size_train) + ((self.train_epochs-1)*len(self.train_data.dataset)))

        return network

    def test(self, network):
        """ Tests network on the test set"""

        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_data:
                output = network(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
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
        plt.ylabel('Loss')
        plt.show()


