from storage_functions import experience_replay_sender, experience_replay_server
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Queue, Process
from training_functions import model_trainer, writer_worker
from go_model import ResNet, ConvResNet, ResNetBasicBlock

import torch.nn as nn
from functools import partial
import torch
import numpy as np
import torch.nn.functional as F
from resnet_model import ResNet
from torchvision.models.resnet import BasicBlock


if __name__ == '__main__':
    MCTS_settings = {"n_parallel_explorations": 4,  # Number of pseudo-parrallel runs of the MCTS, note >16 reduces accuracy significantly
                     "action_size": (4,),  # size of action space
                     "hidden_S_size": (3, 3),  # Size of the hidden state
                     "observation_size": (3,3),  # shape of observation space
                     "gamma": 0.1}  # parameter for pUCT selection

    # Settings for experience replay and storing of values in general
    experience_settings = {"history_size": 50,  # The number of sequences of frames to store in memory
                        "sequence_length": 100,  # The number of frames in each sequence
                        "n_bootstrap": 1,  # Number of steps forward to bootstrap from
                        "past_obs": 6,
                        "K": 1  # Number of steps to unroll during training. Needed here to determine delay of sending
                       }
    training_settings = {"train_batch_size": 32,  # Batch size on GPU during training
                         "num_epochs": 4*10**4,
                         "alpha": 1,
                         "beta": 1,
                         "lr_init": 0.003,  # Original Atari rate was 0.05
                         "lr_decay_rate": 0.5, # Original Atari rate was 0.1
                         "lr_decay_steps": 500,  # Original Atari was 350e3
                         "momentum": 0.9  # Original was 0.9
                         }
    Q_writer = Queue()
    training_settings["Q_writer"] = Q_writer
    experience_settings["Q_writer"] = Q_writer
    MCTS_settings["Q_writer"] = Q_writer

    gamma = MCTS_settings["gamma"]
    obs_size = MCTS_settings["observation_size"]
    action_size = MCTS_settings["action_size"]
    past_obs = experience_settings["past_obs"]

    K = experience_settings["K"]
    ex_Q = Queue()
    EX_server = experience_replay_server(ex_Q, experience_settings, MCTS_settings)
    EX_sender = experience_replay_sender(ex_Q, 1, gamma, experience_settings)
    np.random.seed(1)
    N_episodes = 1
    max_episode_len = 100
    total_samples = 0
    # Add samples to experience replay
    for i in range(N_episodes):
        # Sample epiosde length
        episode_len = 34 #np.random.randint(1, max_episode_len)

        S_stack = np.random.rand(episode_len, 1, obs_size[0], obs_size[1]).astype(np.float32)
        S_stack[:, 0,  :, :] = np.arange(episode_len)[:,None, None]
        #for ii in range(obs_size[0]):
        #    for jj in range(obs_size[1]):
        #        S_stack[:, ii, jj] = np.arange(episode_len)

        a_stack = np.zeros((episode_len))
        r_stack = np.arange(episode_len, dtype=np.float64)  # np.ones((episode_len,))#
        v_stack = np.arange(episode_len, dtype=np.float64)
        done_stack = np.zeros((episode_len,))
        done_stack[-1] = 1
        pi_stack = np.zeros((episode_len, action_size[0]))
        pi_stack[:, 0] = 1

        for j in range(episode_len):
            EX_sender.store(S_stack[j], a_stack[j], r_stack[j], done_stack[j], v_stack[j], pi_stack[j])
            while not ex_Q.empty():
                EX_server.recv_store()
        total_samples += episode_len
    # Catch last values bevause of delay of Queue.empty()
    time.sleep(0.1)
    while not ex_Q.empty():
        EX_server.recv_store()

    """
    def resnet40(in_channels, filter_size=128, board_size=9, deepths=[19]):
        return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)


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
            bs = x.shape[0]
            x_flat = x.view((bs,  ) + (np.prod(self.input_shape),))  # Flatten
            # Hidden state
            S = self.activation1_1(self.layer1_1(x_flat))
            S = self.activation1_2(self.layer1_2(S))
            S = torch.reshape(S, (bs,) + self.output1_shape)  # Residual connection
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
            bs = x.shape[0]
            x_flat = x.view((-1, ) + (np.prod(self.input_shape),))  # Flatten
            S = self.activation1_1(self.layer1_1(x_flat))
            S = self.activation1_2(self.layer1_2(S))  # Residual connection
            S = torch.reshape(S, (bs,1) + self.output1_shape)
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


    class Net(nn.Module):
        def __init__(self, output_shape):
            super().__init__()
            self.output_shape = output_shape
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(2*144, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, np.prod(self.output_shape))

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = x.reshape((x.shape[0],1) + self.output_shape)
            return x

    class TwoNet(nn.Module):
        def __init__(self, input_channel, planes, policy_shape, policy_channels):
            super().__init__()
            self.policy_shape = policy_shape
            self.policy_channels = policy_channels
            self.conv1 = nn.Conv2d(input_channel, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(144*2, 128)
            self.fc2 = nn.Linear(128, 64)
            self.policyHead = nn.Linear(64,  np.prod(self.policy_shape)*self.policy_channels)
            self.valueHead = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            policy = self.policyHead(x)
            value = self.valueHead(x)
            policy = policy.reshape((policy.shape[0], self.policy_channels) + self.policy_shape)
            return policy, value
    """


    class h_resnet(nn.Module):
        def __init__(self):
            super().__init__()
            _COMMON_META = {
                "task": "image_classification",
                "size": (3, 3),
                "min_size": (64, 64),
                "categories": 9,
            }
            self.resnet =  ResNet(BasicBlock, [2, 2, 2, 2], inplanes = 1, num_classes = 9)

        def forward(self, x):
            x = self.resnet(x)
            x = x.reshape((x.shape[0], 1) + (3,3))
            return x

    class g_resnet(nn.Module):
        def __init__(self):
            super().__init__()
            _COMMON_META = {
                "task": "image_classification",
                "size": (3, 3),
                "min_size": (64, 64),
                "categories": 9,
            }
            self.resnet =  ResNet(BasicBlock, [2, 2, 2, 2], inplanes = 5, num_classes = 1+9)

        def forward(self, x):
            x = self.resnet(x)
            [value, policy] = torch.split(x, [1, 9], dim=1)
            policy = policy.reshape((policy.shape[0], 1) + (3,3))
            return policy, value

    class f_resnet(nn.Module):
        def __init__(self):
            super().__init__()
            _COMMON_META = {
                "task": "image_classification",
                "size": (3, 3),
                "min_size": (64, 64),
                "categories": 9,
            }
            self.resnet =  ResNet(BasicBlock, [2, 2, 2, 2], inplanes = 1, num_classes = 1+4)

        def forward(self, x):
            x = self.resnet(x)
            [value, policy] = torch.split(x, [1, 4], dim=1)
            policy = policy.reshape((policy.shape[0],) + MCTS_settings["action_size"])
            return policy, value

    hidden_shape = (3,3)
    """
    import hiddenlayer as hl
    
    transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
    batch = torch.zeros(32,2,2)
    #inter = h_model.forward(batch)
    
    graph = hl.build_graph(f_model, batch, transforms=transforms)
    graph.theme = hl.graph.THEMES['blue'].copy()
    graph.save('rnn_hiddenlayer', format='png')
    """

    torch.multiprocessing.set_start_method('spawn', force=True)
    wr_worker = Process(target=writer_worker, args=(Q_writer,))
    wr_worker.start()

    lr_list = torch.linspace(0, -3, 5)
    lr_list = 10**lr_list


    min_vals = []
    for lr in lr_list:
        f_model = f_resnet() #dummy_networkF(hidden_shape, action_size, 4)
        g_model = g_resnet()#TwoNet(5, 32, hidden_shape, 1)  # dummy_networkG((5,)+hidden_shape, (1,)+hidden_shape, 64)
        h_model = h_resnet()
        trainer = model_trainer(f_model, g_model, h_model, EX_server, experience_settings, training_settings, MCTS_settings)
        trainer.train()



