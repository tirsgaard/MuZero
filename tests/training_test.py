from storage_functions import experience_replay_sender, experience_replay_server
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from training_functions import model_trainer
from go_model import ResNet, ConvResNet, ResNetBasicBlock

MCTS_settings = {"n_parallel_explorations": 4,  # Number of pseudo-parrallel runs of the MCTS, note >16 reduces accuracy significantly
                 "action_size": (4,),  # size of action space
                 "observation_size": (3,3),  # shape of observation space
                 "gamma": 0.9}  # parameter for pUCT selection

# Settings for experience replay and storing of values in general
experience_settings = {"history_size": 50,  # The number of sequences of frames to store in memory
                    "sequence_length": 10,  # The number of frames in each sequence
                    "n_bootstrap": 4,  # Number of steps forward to bootstrap from
                    "past_obs": 1,
                    "K": 1  # Number of steps to unroll during training. Needed here to determine delay of sending
                   }
training_settings = {"train_batch_size": 48,  # Batch size on GPU during training
                     "num_epochs": 2000,
                     "alpha": 1,
                     "lr_init": 0.05,  # Original Atari rate was 0.05
                     "lr_decay_rate": 0.1,
                     "lr_decay_steps": 400e3,  # Original Atari was 350e3
                     "momentum": 0.0  # Original was 0.9
                     }


gamma = MCTS_settings["gamma"]
obs_size = MCTS_settings["observation_size"]
action_size = MCTS_settings["action_size"]
past_obs = experience_settings["past_obs"]

K = experience_settings["K"]
EX_server = experience_replay_server(experience_settings, MCTS_settings)
get_ex_Q = EX_server.get_Q()
EX_sender = experience_replay_sender(get_ex_Q, 1, gamma, experience_settings)
np.random.seed(1)
N_episodes = 50
max_episode_len = 100
total_samples = 0
# Add samples to experience replay
for i in range(N_episodes):
    # Sample epiosde length
    episode_len = np.random.randint(1, max_episode_len)

    S_stack = np.zeros((episode_len, obs_size[0], obs_size[1]))
    for ii in range(obs_size[0]):
        for jj in range(obs_size[1]):
            S_stack[:, ii, jj] = np.arange(episode_len)

    a_stack = np.zeros((episode_len, action_size[0]))
    r_stack = np.arange(episode_len)  # np.ones((episode_len,))#
    v_stack = np.arange(episode_len)
    done_stack = np.zeros((episode_len,))
    done_stack[episode_len-1] = 1
    pi_stack = np.zeros((episode_len, action_size[0]))
    pi_stack[:, 0] = 1

    for j in range(episode_len):
        EX_sender.store(S_stack[j], a_stack[j], r_stack[j], done_stack[j], v_stack[j], pi_stack[j])
        while not get_ex_Q.empty():
            EX_server.recv_store()
    total_samples += episode_len

writer = SummaryWriter()
trainer = model_trainer(writer, EX_server, experience_settings, training_settings, MCTS_settings)


def resnet40(in_channels, filter_size=128, board_size=9, deepths=[19]):
    return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)

import torch.nn as nn
from functools import partial
import torch
import numpy as np
import torch.nn.functional as F


class dummy_networkG(nn.Module):
    def __init__(self, input_shape, output1_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.layer1 = nn.Linear(np.prod(input_shape), np.prod(output1_shape), bias=True)
        self.activation1 = nn.ELU()
        self.activation2 = nn.ELU()
        self.layer2 = nn.Linear(np.prod(input_shape), 1, bias=True)


    def forward(self, x):
        x = torch.reshape(x, (-1, ) + (np.prod(self.input_shape),)) # Flatten
        S = self.activation1(self.layer1(x))
        S = torch.reshape(S, (-1,) + self.output1_shape)
        reward = self.activation2(self.layer2(x))
        return [S, reward]

class dummy_networkH(nn.Module):
    def __init__(self, input_shape, output1_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape
        self.layer1 = nn.Linear(np.prod(input_shape), np.prod(output1_shape), bias=True)
        self.activation1 = nn.ELU()

    def forward(self, x):
        x = torch.reshape(x, (-1, ) + (np.prod(self.input_shape),))  # Flatten
        S = self.activation1(self.layer1(x))
        S = torch.reshape(S, (-1,) + self.output1_shape)
        return S

class dummy_networkF(nn.Module):
    def __init__(self, input_shape, output1_shape):
        super(dummy_networkF, self).__init__()
        self.input_shape = input_shape
        self.output1_shape = output1_shape

        hidden_n = 256
        self.layer1_1 = nn.Linear(np.prod(input_shape), hidden_n, bias=True)
        self.layer1_2 = nn.Linear(hidden_n, np.prod(output1_shape), bias=True)
        self.activation1_1 = nn.ELU()
        self.activation1_2 = nn.Softmax(dim=1)
        self.activation2_1 = nn.ELU()

        self.layer2_1 = nn.Linear(np.prod(input_shape), hidden_n, bias=True)
        self.layer2_2 = nn.Linear(hidden_n, 1, bias=False)

    def forward(self, x):
        x = torch.reshape(x, (-1, ) + (np.prod(self.input_shape),))  # Flatten
        policy = F.relu(self.layer1_1(x))
        policy = self.activation1_2(self.layer1_2(policy))
        policy = torch.reshape(policy, (-1,) + self.output1_shape)


        value = F.relu(self.layer2_1(x))
        value2 = F.relu(value)
        return [policy, value2]


f_model = dummy_networkF(obs_size, action_size) #ResNet(64, 64, 9, (action_size[0], ), block=ResNetBasicBlock, deepths=[1])
g_model = dummy_networkG(obs_size, obs_size)#ResNet(64, 64, 9, (64, 3, 3), block=ResNetBasicBlock, deepths=[1])
h_model = dummy_networkH((past_obs,) + obs_size, obs_size)#ConvResNet(obs_size, obs_size)
trainer.train(f_model, g_model, h_model)