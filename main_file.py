#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:44:33 2019

@author: tirsgaard
"""
# File for running the entire program

#from go_model import ResNet, ResNetBasicBlock
from game_functions import sim_games
from training_functions import save_model, load_latest_model, model_trainer, train_ex_worker, writer_worker
from torch.multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool, SimpleQueue
from storage_functions import experience_replay_server
from models import dummy_networkF, dummy_networkG, dummy_networkH
from go_model import ResNet_f, ResNet_g, ConvResNet
from models import identity_networkH, identity_networkF, identity_networkG, constant_networkF
import gym
import hyperparameters as conf

import torch

if __name__ == '__main__':

    # Get configs
    MuZero_settings = conf.MuZero_settings
    experience_settings = conf.experience_settings
    MCTS_settings = conf.MCTS_settings
    training_settings = conf.training_settings
    # Construct networks
    hidden_shape = (1, ) + MCTS_settings["hidden_S_size"]
    action_size = MCTS_settings["action_size"]
    hidden_input_size = (MCTS_settings["action_size"][0] + 1,) + MCTS_settings["hidden_S_size"]

    f_model = dummy_networkF(hidden_shape, action_size,
                             32)  # Model for predicting value (v) and policy (p)
    g_model = dummy_networkG(hidden_input_size, hidden_shape, 32)  # Model for predicting hidden state (S)
    h_model = dummy_networkH((experience_settings["past_obs"],) + MCTS_settings["observation_size"], hidden_shape,
                             32)  # Model for converting environment state to hidden state


    #h_model = ConvResNet(experience_settings["past_obs"], MCTS_settings["hidden_S_channel"], hidden_shape)  # identity_networkH((1, 2, 2), hidden_shape)
    #g_model = ResNet_g(MCTS_settings["hidden_S_channel"]+action_size[0], 32, hidden_shape, MCTS_settings["hidden_S_channel"], 128)  # identity_networkG(hidden_input_size, hidden_shape)
    #f_model = identity_networkF(hidden_shape, action_size)  # ResNet_f(MCTS_settings["hidden_S_channel"], 32, 4, action_size, 128)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    f_model.share_memory()
    g_model.share_memory()
    h_model.share_memory()

    # Function for creating environment. Needs to create seperate env for each worker
    class RewardWrapper(gym.RewardWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.iters = 0

        def reward(self, rew):
            # modify rew
            rew = rew * (self.iters % 2)
            self.iters += 1
            return rew

    env_maker = lambda: RewardWrapper(gym.make("CartPole-v1"))

    # Construct model trainer and experience storage
    torch.multiprocessing.set_start_method('spawn', force=True)
    Q_writer = Queue()
    training_settings["Q_writer"] = Q_writer
    experience_settings["Q_writer"] = Q_writer
    MCTS_settings["Q_writer"] = Q_writer
    ER_Q = Queue()
    ER_worker = Process(target=train_ex_worker, args=(ER_Q, f_model, g_model, h_model, experience_settings, training_settings, MCTS_settings))
    ER_worker.start()

    # Worker for storing statistics
    torch.multiprocessing.set_start_method('fork', force=True)
    wr_worker = Process(target=writer_worker, args=(Q_writer,))
    wr_worker.start()
    # define variables to be used
    loop_counter = 1
    training_counter = 0
    # Running loop
    print("Beginning loop", loop_counter)
    print("Beginning self-play")
    # Generate new data for training
    with torch.no_grad():
        sim_games(env_maker, f_model, g_model, h_model, ER_Q, MCTS_settings, MuZero_settings, experience_settings)

    loop_counter += 1
