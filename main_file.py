#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:44:33 2019

@author: tirsgaard
"""
# File for running the entire program

from go_model import ResNet, ResNetBasicBlock
from game_functions import sim_games
from training_functions import save_model, load_latest_model, model_trainer
from storage_functions import experience_replay_server
from models import dummy_networkF, dummy_networkG, dummy_networkH
import gym
import hyperparameters as conf

import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    def resnet40(in_channels, filter_size=128, board_size=9, deepths=[19]):
        return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)

    # Get configs
    MuZero_settings = conf.MuZero_settings
    experience_settings = conf.experience_settings
    MCTS_settings = conf.MCTS_settings
    training_settings = conf.training_settings
    # Construct networks
    hidden_shape = MCTS_settings["hidden_S_size"]
    f_model = dummy_networkF(hidden_shape, MCTS_settings["action_size"], 32)  # Model for predicting value (v) and policy (p)
    g_model = dummy_networkG(hidden_shape, hidden_shape, 32)  # Model for predicting hidden state (S)
    h_model = dummy_networkH((experience_settings["past_obs"],) + MCTS_settings["observation_size"], hidden_shape, 32)  # Model for converting environment state to hidden state

    # GPU things
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
    if cuda:
        f_model.cuda()
        g_model.cuda()
        h_model.cuda()

    # Construct model trainer and experience storage
    writer = SummaryWriter()
    ER = experience_replay_server(experience_settings, MCTS_settings)
    ER_Q = ER.get_Q()
    trainer = model_trainer(writer, ER, experience_settings, training_settings, MCTS_settings)
    env_maker = lambda: gym.make("CartPole-v1")  # Function for creating environment. Needs to create seperate env for each worker

    # define variables to be used
    loop_counter = 1
    training_counter = 0
    # Running loop
    while True:
        print("Beginning loop", loop_counter)
        print("Beginning self-play")
        # Generate new data for training
        with torch.no_grad():
            sim_games(env_maker, f_model, g_model, h_model, ER_Q, MCTS_settings, MuZero_settings)

        print("Begin training")
        # Now train model
        training_model = trainer.train(training_model)

        #save_model(f_model)
        #save_model(g_model)
        #save_model(h_model)
        loop_counter += 1
