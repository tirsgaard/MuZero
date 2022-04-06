import torch.nn as nn
from functools import partial
import torch
import numpy as np
import torch.nn.functional as F


def stack_a(S, a, hidden_shape, action_size):
    a_onehot = np.zeros(action_size + hidden_shape)
    a_onehot[a, : , :] = 1  # One hot plane
    S = np.concatenate([S, a_onehot], axis=0)
    return S

def stack_a_torch(S, a, hidden_shape, action_size):
    batch_size = a.shape[0]
    a_onehot = torch.zeros((batch_size,) + action_size + hidden_shape)
    a_onehot[range(batch_size), a, : , :] = 1  # One hot plane
    Sa = torch.cat([S, a_onehot], dim=1)
    return Sa

class dummy_networkG(nn.Module):
    def __init__(self, input_shape, output1_shape, hidden_size):
        super().__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.hidden_size = hidden_size

        # Hidden state head
        self.layer1_1 = nn.Linear(np.prod(input_shape), self.hidden_size)
        self.activation1_1 = nn.ReLU()
        self.layer1_2 = nn.Linear(self.hidden_size, np.prod(output1_shape))
        self.activation1_2 = nn.ReLU()
        # Reward head
        self.layer2_1 = nn.Linear(np.prod(input_shape), self.hidden_size)
        self.activation2_1 = nn.ReLU()
        self.layer2_2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x_flat = x.view((-1, ) + (np.prod(self.input_shape),))  # Flatten
        # Hidden state
        S = self.activation1_1(self.layer1_1(x_flat))
        S = self.activation1_2(self.layer1_2(S))
        S = torch.reshape(S, (-1,) + self.output1_shape)
        # Reward state
        reward = self.activation2_1(self.layer2_1(x_flat))
        reward = self.layer2_2(reward)
        return [S, reward]

class dummy_networkH(nn.Module):
    def __init__(self, input_shape, output1_shape, hidden_size):
        super().__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.hidden_size = hidden_size
        self.layer1_1 = nn.Linear(np.prod(input_shape), self.hidden_size)
        self.activation1_1 = nn.ReLU()
        self.layer1_2 = nn.Linear(self.hidden_size, np.prod(output1_shape))
        self.activation1_2 = nn.ReLU()

    def forward(self, x):
        x_flat = x.view((-1, ) + (np.prod(self.input_shape),))  # Flatten
        S = self.activation1_1(self.layer1_1(x_flat))
        S = self.activation1_2(self.layer1_2(S))
        S = torch.reshape(S, (-1,) + self.output1_shape)
        return S

class dummy_networkF(nn.Module):
    def __init__(self, input_shape, output1_shape, hidden_size):
        super(dummy_networkF, self).__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.hidden_size = hidden_size
        # Policy head
        self.layer1_1 = nn.Linear(np.prod(input_shape), self.hidden_size)
        self.activation1_1 = nn.ReLU()
        self.layer1_2 = nn.Linear(self.hidden_size, np.prod(output1_shape))
        self.activation1_2 = nn.Softmax(dim=1)
        # Value head
        self.layer2_1 = nn.Linear(np.prod(input_shape), self.hidden_size)
        self.activation2_1 = nn.ReLU()
        self.layer2_2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x_flat = x.view((-1, ) + (np.prod(self.input_shape),) )  # Flatten
        # Policy head
        policy = self.activation1_1(self.layer1_1(x_flat))
        policy = self.activation1_2(self.layer1_2(policy))
        policy = torch.reshape(policy, (-1,) + self.output1_shape)  # Residual connection
        # Value head
        value = self.activation2_1(self.layer2_1(x_flat))
        value = self.layer2_2(value)
        return [policy, value]