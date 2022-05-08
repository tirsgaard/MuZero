import torch.nn as nn
from functools import partial
import torch
import numpy as np
import torch.nn.functional as F


class oracleH(nn.Module):
    def __init__(self):
        super(oracleH, self).__init__()

    def forward(self, x):
        return x.reshape(-1, 1, 3, 3)


def bin_2_dec(bin):
    # Assumes input shape  is (-1, 7)
    mask = 2**torch.tensor([6, 5, 4, 3, 2, 1, 0])
    dec = mask[None, :]*bin
    dec = dec.sum(dim=1)
    return dec


def dec_2_bin(x, bits):
    mask = 2**torch.arange(bits-1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class oracleG(nn.Module):
    def __init__(self):
        super(oracleG, self).__init__()

    def forward(self, x):
        S = x[:, 0, :, :].clone()
        old_step = S[:, 2, 0]
        action = x[:, 1, 0, 0] != 0  # Is this action 0
        reward = old_step != action
        loose_life = (~reward).to(torch.long)

        # Convert from binary to decimal
        old_step = bin_2_dec(S.reshape((-1, 9))[:, 0:7])
        reward = reward * (old_step < 101)
        bin = dec_2_bin(old_step.to(torch.long) + 1, 7)
        S = S.reshape(-1,9)
        S[:, 0:7] = bin
        S[:, 7] = S[:, 7] - loose_life
        S[:, 7] = (S[:, 7] > 0)*S[:, 7]  # Keep non-negative
        S = S.reshape(-1, 1, 3, 3)
        return [S, reward[:,None].to(torch.float32)]


class half_oracleF(nn.Module):
    def __init__(self, input_shape, output1_shape, hidden_size):
        super(half_oracleF, self).__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.hidden_size = hidden_size
        self.train_count = 0
        # Policy head
        self.layer1_1 = nn.Linear(np.prod(input_shape), self.hidden_size)
        self.activation1_1 = nn.ReLU()
        self.layer1_2 = nn.Linear(self.hidden_size, np.prod(output1_shape))
        self.activation1_2 = nn.Softmax(dim=1)
        # Value head

    def forward(self, x):
        step = bin_2_dec(x.reshape((-1, 9))[:, 0:7])
        v = (100 - step) * ((x[:, 0, 2, 1]>0) * (step<100))
        x_flat = x.view((-1, ) + (np.prod(self.input_shape),) )  # Flatten
        x_flat = x_flat.to(torch.float)
        # Policy head
        policy = self.activation1_1(self.layer1_1(x_flat))
        policy = self.activation1_2(self.layer1_2(policy))
        policy = torch.reshape(policy, (-1,) + self.output1_shape)  # Residual connection
        # Value head
        return [policy, v[:, None]]

