from storage_functions import experience_replay_sender, experience_replay_server
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Queue, Process
from training_functions import model_trainer, writer_worker
from go_model import ResNet, ConvResNet, ResNetBasicBlock, ResNet_g

import torch.nn as nn
from functools import partial
import torch
import numpy as np
import torch.nn.functional as F
from resnet_model import ResNet
from torchvision.models.resnet import BasicBlock
from models import identity_networkH, identity_networkF, dummy_networkF, dummy_networkH, ram_network_convG, ram_network_convH, ram_network_convF


if __name__ == '__main__':
    MuZero_settings = {"N_training_games": 200000,  # Total number of games to run pr. training loop
                       "temp_switch": 16,  # Number of turns before other temperature measure is used
                       "eta_par": 0.03,  # Distributional value for action selection
                       "epsilon": 0.25,  # Distributional value for action selection
                       "save_image": False,  # Save image of environment when MCT is saved
                       "low_support": -50,  # Lowest value of supported values for reward and value head
                       "high_support": 400,  # Highest value of supported values for reward and value head
                       "n_support": 451,
                       # Number of support values. To include a value for 0 keep the number of heads odd
                       }

    MCTS_settings = {"n_parallel_explorations": 4,  # Number of pseudo-parrallel runs of the MCTS, note >16 reduces accuracy significantly
                     "action_size": (4,),  # size of action space
                     "observation_size": (2, 2),  # shape of observation space
                     "observation_channels": 1,  # number of channels of observation space (i.e. 3*4 for RGB and 4x frame stack)
                     "hidden_S_size": (3, 3),  # Size of the hidden state
                     "hidden_S_channel": 1,  # Size of the hidden state
                     "gamma": 0.1}  # parameter for pUCT selection

    # Settings for experience replay and storing of values in general
    experience_settings = {"history_size": 500,  # The number of sequences of frames to store in memory
                        "sequence_length": 1000,  # The number of frames in each sequence
                        "n_bootstrap": 1,  # Number of steps forward to bootstrap from
                        "past_obs": 1,
                        "K": 10  # Number of steps to unroll during training. Needed here to determine delay of sending
                       }
    training_settings = {"train_batch_size": 256,  # Batch size on GPU during training
                         "num_epochs": 1000,
                         "alpha": 1,
                         "beta": 1,
                         "lr_init": 1.*10**-8,  # Original Atari rate was 0.05
                         "lr_decay_rate": 0.5, # Original Atari rate was 0.1
                         "lr_decay_steps": 10000,  # Original Atari was 350e3
                         "momentum": 0.9,  # Original was 0.9
                         "weight_decay": 1e-4,
                         "uniform_sampling": False,
                         "scale_values": True,
                         }
    torch.manual_seed(0)
    np.random.seed(1)
    Q_writer = Queue()
    training_settings["Q_writer"] = Q_writer
    experience_settings["Q_writer"] = Q_writer
    MCTS_settings["Q_writer"] = Q_writer

    gamma = MCTS_settings["gamma"]
    obs_size = MCTS_settings["observation_size"]
    action_size = MCTS_settings["action_size"]
    past_obs = experience_settings["past_obs"]
    observation_shape = (experience_settings["past_obs"] * MCTS_settings["observation_channels"],) + MCTS_settings[
        "observation_size"]  # input to f
    hidden_shape = (MCTS_settings["hidden_S_channel"],) + MCTS_settings["hidden_S_size"]  # input to f
    hidden_input_size = (MCTS_settings["action_size"][0] + MCTS_settings["hidden_S_channel"],) + MCTS_settings[
        "hidden_S_size"]  # Input to g
    n_heads = MuZero_settings["n_support"]
    transform_values = training_settings["scale_values"]


    K = experience_settings["K"]
    ex_Q = Queue()
    EX_server = experience_replay_server(ex_Q, experience_settings, MCTS_settings)
    EX_sender = experience_replay_sender(ex_Q, 1, gamma, experience_settings)

    N_episodes = 10**3
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

        a_stack = np.random.randint(0,np.prod(action_size), episode_len)
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

    hidden_shape = (1,3,3)

    torch.multiprocessing.set_start_method('spawn', force=True)
    wr_worker = Process(target=writer_worker, args=(Q_writer,))
    wr_worker.start()
    n_heads = MuZero_settings["n_support"]
    support = torch.linspace(MuZero_settings["low_support"], MuZero_settings["high_support"], n_heads)

    f_model = ram_network_convF(hidden_shape, MCTS_settings["action_size"], 1024, support, transform_values)
    g_model = ram_network_convG(hidden_input_size, hidden_shape, 1024, support, transform_values)
    h_model = ram_network_convH(observation_shape, hidden_shape, 128)
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
    print("begin fetching")
    time_start = time.time()
    for i in range(100):
        print(i)
        EX_server.return_batches(4096,
                               1,
                               10,
                               uniform_sampling=False)
    time_end = time.time()
    duration = time_end - time_start
    print("Iterations pr. sec:" + str(round(100 / duration)))
    trainer.train()
    wr_worker.terminate()


