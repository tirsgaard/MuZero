import torch.nn as nn
from functools import partial
import torch
import numpy as np
import torch.nn.functional as F
cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

def stack_a(S, a, hidden_shape, action_size):
    a_onehot = np.zeros((1, ) + action_size + hidden_shape)
    a_onehot[0, a, :, :] = 1/np.prod(action_size)  # One hot plane
    S = np.concatenate([S[None], a_onehot], axis=1)
    return S

def stack_a_torch(S, a, hidden_shape, action_size):
    batch_size = a.shape[0]
    a_onehot = torch.zeros((batch_size,) + action_size + hidden_shape)
    a_onehot[range(batch_size), a, :, :] = 1/np.prod(action_size)  # One hot plane
    Sa = torch.cat([S, a_onehot], dim=1)
    return Sa

class dummy_networkG(nn.Module):
    def __init__(self, input_shape, output1_shape, hidden_size):
        super().__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.hidden_size = hidden_size
        self.train_count = 0

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
        # Add skip connection
        S = S + x[:, 0, None]

        # Reward state
        reward = self.activation2_1(self.layer2_1(x_flat))
        reward = self.layer2_2(reward)
        reward = reward
        return [S, reward]

class dummy_networkH(nn.Module):
    def __init__(self, input_shape, output1_shape, hidden_size):
        super().__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.hidden_size = hidden_size
        self.train_count = 0
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
        self.train_count = 0
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
        value = value
        return [policy, value]

class constant_networkF(nn.Module):
    def __init__(self, input_shape, output1_shape, hidden_size):
        super(constant_networkF, self).__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.hidden_size = hidden_size
        self.train_count = 0
        # Policy head
        # Value head
        self.layer2_1 = nn.Linear(np.prod(input_shape), self.hidden_size)
        self.activation2_1 = nn.ReLU()
        self.layer2_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation2_2 = nn.ReLU()
        self.layer2_3 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        bs = x.shape[0]
        x_flat = x.view((-1, ) + (np.prod(self.input_shape),) )  # Flatten
        # Policy head
        policy = torch.ones((bs,) + self.output1_shape)/np.prod(self.output1_shape)  # Residual connection
        # Value head
        value = self.activation2_1(self.layer2_1(x_flat))
        value = self.activation2_2(self.layer2_2(value))
        value = self.layer2_3(value)
        return [policy, value]


class identity_networkF(nn.Module):
    def __init__(self, input_shape, output1_shape):
        super(identity_networkF, self).__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.train_count = 0
        # Policy head
        self.layer1_1 = nn.Linear(np.prod(input_shape), np.prod(output1_shape))
        # Value head
        self.layer2_1 = nn.Linear(np.prod(input_shape), 1)
        self.softmaxer = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x_flat = x.view((-1, ) + (np.prod(self.input_shape),) )  # Flatten
        # Policy head
        policy = self.layer1_1(x_flat)
        policy = torch.reshape(policy, (-1,) + self.output1_shape)  # Residual connection
        policy = self.softmaxer(policy)  # Normaliser
        # Value head
        value = self.layer2_1(x_flat)
        return [policy, value]


class identity_networkH(nn.Module):
    def __init__(self, input_shape, output1_shape):
        super(identity_networkH ,self).__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.train_count = 0
        self.layer1_1 = nn.Linear(np.prod(input_shape), np.prod(output1_shape))
        self.activation1_1 = nn.ReLU()

    def forward(self, x):
        x_flat = x.view((-1, ) + (np.prod(self.input_shape),))  # Flatten
        S = self.activation1_1(self.layer1_1(x_flat))
        S = torch.reshape(S, (-1,) + self.output1_shape)
        return S


class identity_networkG(nn.Module):
    def __init__(self, input_shape, output1_shape):
        super(identity_networkG, self).__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.train_count = 0

        # Hidden state head
        self.layer1_1 = nn.Linear(np.prod(input_shape), np.prod(output1_shape))
        self.activation1_1 = nn.ReLU()
        # Reward head
        self.layer2_1 = nn.Linear(np.prod(input_shape), 1)

    def forward(self, x):
        x_flat = x.view((-1, ) + (np.prod(self.input_shape),))  # Flatten
        # Hidden state
        S = self.activation1_1(self.layer1_1(x_flat))
        S = torch.reshape(S, (-1,) + self.output1_shape)

        # Reward state
        reward = self.layer2_1(x_flat)
        return [S, reward]

class oracleH(nn.Module):
    def __init__(self):
        super(oracleH, self).__init__()

    def forward(self, x):
        S = x
        return torch.reshape(S, (-1, 1, 2, 2))

class oracleG(nn.Module):
    def __init__(self):
        super(oracleG, self).__init__()

    def forward(self, x):
        S = x[:, 0].clone()
        old_step = S[:, 0, 0]
        action = x[:, 1, 0, 0] != 0  # Is this action 0
        S[:, 0, 0] += 1
        reward = old_step % 2 == action
        S[:, 0, 1] -= (~reward).to(torch.long)
        return [S[:, None], reward[:, None].to(torch.float32)]

class half_oracleG(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(oracleG, self).__init__()
        self.hidden_size = hidden_size
        # Reward head
        self.layer2_1 = nn.Linear(np.prod(input_shape), self.hidden_size)
        self.activation2_1 = nn.ReLU()
        self.layer2_2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        S = x[:, 0].clone()
        old_step = S[:, 0, 0]
        action = x[:, 1, 0, 0] != 0  # Is this action 0
        S[:, 0, 0] += 1
        reward = old_step % 2 == action
        S[:, 0, 1] -= (~reward).to(torch.long)

        # Reward state
        x_flat = x.view((-1,) + (np.prod(self.input_shape),))  # Flatten
        reward = self.activation2_1(self.layer2_1(x_flat))
        reward = self.layer2_2(reward)
        reward = reward
        return [S[:, None], reward]

class oracleF(nn.Module):
    def __init__(self):
        super(oracleF, self).__init__()

    def forward(self, x):
        step = x[:, 0, 0, 0]
        v = 100 - step
        best_action = (step % 2).to(torch.long)
        P = torch.ones(x.shape[0], 2)*0.2
        P[:, best_action] = 0.8
        return [P, v[:, None]]


class muZero(nn.Module):
    def __init__(self, f_model, g_model, h_model, K, hidden_S_size, action_size):
        super(muZero, self).__init__()
        self.f_model = f_model
        self.g_model = g_model
        self.h_model = h_model
        self.K = K
        self.hidden_S_size = hidden_S_size
        self.action_size = action_size

    def forward(self, S, a_batch, z_batch):
        p_vals = []  # Number
        r_batches = []
        v_batches = []
        P_batches = []

        new_S = self.h_model.forward(S[:, -1])  # Only the most recent of the unrolled observations are used
        for k in range(self.K):
            P_batch, v_batch = self.f_model.forward(new_S)
            Sa_batch = stack_a_torch(new_S, a_batch[:, k], self.hidden_S_size, self.action_size)
            new_S, r_batch = self.g_model.forward(Sa_batch)
            p_vals.append(torch.abs(v_batch.squeeze(dim=1) - z_batch[:, k]).detach().cpu().numpy())  # For importance weighting
            P_batches.append(P_batch)
            v_batches.append(v_batch)
            r_batches.append(r_batch)

        P_batches = torch.stack(P_batches, dim=1)
        v_batches = torch.stack(v_batches, dim=1).squeeze(dim=2)
        r_batches = torch.stack(r_batches, dim=1).squeeze(dim=2)
        p_vals = np.stack(p_vals, axis=1)

        return P_batches, v_batches, r_batches, p_vals

def h_scale(x, epsilon = 0.01):
    y = torch.sign(x)*(torch.sqrt(torch.abs(x)+1)-1)+epsilon*x
    return y

def h_inverse_scale(y, epsilon = 0.01):
    intermid = (torch.sqrt(1+4*epsilon*(torch.abs(y)+1+epsilon))-1)/(2*epsilon)
    x = torch.sign(y)*(intermid*intermid-1)
    return x