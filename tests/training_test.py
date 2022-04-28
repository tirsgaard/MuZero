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
from models import identity_networkH, identity_networkF, identity_networkG
from torchvision.models.resnet import BasicBlock


if __name__ == '__main__':
    MCTS_settings = {"n_parallel_explorations": 4,  # Number of pseudo-parrallel runs of the MCTS, note >16 reduces accuracy significantly
                     "action_size": (4,),  # size of action space
                     "hidden_S_size": (3, 3),  # Size of the hidden state
                     "observation_size": (3,3),  # shape of observation space
                     "gamma": 0.1}  # parameter for pUCT selection

    # Settings for experience replay and storing of values in general
    experience_settings = {"history_size": 256+1,  # The number of sequences of frames to store in memory
                        "sequence_length": 1000,  # The number of frames in each sequence
                        "n_bootstrap": 1,  # Number of steps forward to bootstrap from
                        "past_obs": 6,
                        "K": 3  # Number of steps to unroll during training. Needed here to determine delay of sending
                       }
    training_settings = {"train_batch_size": 256,  # Batch size on GPU during training
                         "num_epochs": 4*10**4,
                         "alpha": 1,
                         "beta": 1,
                         "lr_init": 1.*10**-8,  # Original Atari rate was 0.05
                         "lr_decay_rate": 0.5, # Original Atari rate was 0.1
                         "lr_decay_steps": 10000,  # Original Atari was 350e3
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
    N_episodes = 10
    max_episode_len = 100
    total_samples = 0
    # Add samples to experience replay
    for i in range(N_episodes):
        # Sample epiosde length
        episode_len = 1*256+1 #np.random.randint(1, max_episode_len)

        S_stack = np.random.rand(episode_len, 1, obs_size[0], obs_size[1]).astype(np.float32)
        S_stack[:, 0,  0, 0] = np.arange(episode_len)
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

    class h_resnet(nn.Module):
        def __init__(self):
            super().__init__()
            _COMMON_META = {
                "task": "image_classification",
                "size": (3, 3),
                "min_size": (64, 64),
                "categories": 9,
            }
            self.resnet =  ResNet(BasicBlock, [3, 3, 3, 3], inplanes = 1, num_classes = 9)

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
            self.resnet =  ResNet(BasicBlock, [3, 3, 3, 3], inplanes = 1, num_classes = 1+9)

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
            self.resnet =  ResNet(BasicBlock, [3, 3, 3, 3], inplanes = 1, num_classes = 1+4)

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

    f_model = identity_networkF((1,3,3), (4,)) # dummy_networkF(hidden_shape, action_size, 4)
    g_model = identity_networkG((1,3,3), (1,3,3))#g_resnet()  # TwoNet(5, 32, hidden_shape, 1)  # dummy_networkG((5,)+hidden_shape, (1,)+hidden_shape, 64)
    h_model = identity_networkH((1,3,3), (1,3,3))
    trainer = model_trainer(f_model, g_model, h_model, EX_server, experience_settings, training_settings, MCTS_settings)

    # GPU things
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
    if cuda:
        f_model.to(device)
        g_model.to(device)
        h_model.to(device)

    trainer.train()



