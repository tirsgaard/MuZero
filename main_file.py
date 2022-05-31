#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:44:33 2019

@author: tirsgaard
"""
# File for running the entire program

#from go_model import ResNet, ResNetBasicBlock
from game_functions import sim_games
from training_functions import train_ex_worker, writer_worker
from torch.multiprocessing import Process, Queue
from models import dummy_networkF, dummy_networkG, dummy_networkH
from wrappers import MaxAndSkipEnv, RAMAndSkipEnv
from go_model import ResNet_g, ResNet_f, ResNet_h
import gym
import hyperparameters as conf
from test_env import binTestEnv
import numpy as np

import torch

# TODO remove late start

if __name__ == '__main__':
    # Get configs
    MuZero_settings = conf.MuZero_settings
    experience_settings = conf.experience_settings
    MCTS_settings = conf.MCTS_settings
    training_settings = conf.training_settings
    # Construct networks
    observation_shape = (experience_settings["past_obs"]*MCTS_settings["observation_channels"],) + MCTS_settings["observation_size"]  # input to f
    hidden_shape = (MCTS_settings["hidden_S_channel"], ) + MCTS_settings["hidden_S_size"]  # input to f
    action_size = MCTS_settings["action_size"]
    hidden_input_size = (MCTS_settings["action_size"][0] + MCTS_settings["hidden_S_channel"],) + MCTS_settings["hidden_S_size"]  # Input to g
    n_heads = MuZero_settings["n_support"]
    torch.manual_seed(0)
    np.random.seed(1)

    support = torch.linspace(MuZero_settings["low_support"], MuZero_settings["high_support"], n_heads)
    f_model = dummy_networkF(hidden_shape, action_size, 256, support)
    #in_channels, filter_size, policy_output_shape, output_shape, support
        #dummy_networkF(hidden_shape, action_size, 256, support)

    g_model = ResNet_g(hidden_input_size[0], 256, MCTS_settings["hidden_S_size"],
                       MCTS_settings["hidden_S_channel"], 4096,
                       support)  # half_oracleG((3,3,3), 32) #oracleG() #dummy_networkG(hidden_input_size, hidden_shape, 32)  # Model for predicting hidden state (S)
    h_model = dummy_networkH(observation_shape, hidden_shape, 256)
        #dummy_networkH(observation_shape, hidden_shape, 256)


    #h_model = ConvResNet(experience_settings["past_obs"], MCTS_settings["hidden_S_channel"], hidden_shape)  # identity_networkH((1, 2, 2), hidden_shape)
    #g_model = ResNet_g(MCTS_settings["hidden_S_channel"]+action_size[0], 32, hidden_shape, MCTS_settings["hidden_S_channel"], 128)  # identity_networkG(hidden_input_size, hidden_shape)
    #f_model = identity_networkF(hidden_shape, action_size)  # ResNet_f(MCTS_settings["hidden_S_channel"], 32, 4, action_size, 128)


    # Add mean pass and support
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

    env_maker = lambda: RAMAndSkipEnv(gym.make("ALE/Breakout-v5", full_action_space=False, obs_type="ram"))

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
