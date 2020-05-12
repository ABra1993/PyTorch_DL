import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """ nn.Module is a base class for all neural network modules. """
    def __init__(self):
        """ The super function returns a temporary object that allows reference to a parent class. """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        """ Computation forward step"""
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
