#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:44:33 2019

@author: tirsgaard
"""
# File for running the entire program

from model.go_model import ResNet, ResNetBasicBlock
from MCTS.MCTS2 import sim_games
from tools.training_functions import save_model, load_latest_model, model_trainer

import torch
from torch.utils.tensorboard import SummaryWriter
from tools.elo import elo_league

if __name__ == '__main__':
    def resnet40(in_channels, filter_size=128, board_size=9, deepths=[19]):
        return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)


    f_model = # Model for predicting value (v) and policy (p)
    g_model = # Model for predicting hidden state (S)
    h_model = # Model for converting environment state to hidden state

    elo_league = elo_league()


    # GPU things
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    ## Load model if one exists, else define a new
    writer = SummaryWriter()
    best_model = load_latest_model()
    training_model = load_latest_model()

    if (best_model == None):
        best_model = resnet40(17, 128, board_size=board_size)
        save_model(best_model)
        training_model = load_latest_model()

    if cuda:
        best_model.cuda()
        training_model.cuda()
    dummy_model = resnet40(17, 128, board_size=board_size) # TODO clean this mess up
    trainer = model_trainer(writer, MCTS_settings, training_settings, dummy_model)

    ## define variables to be used
    v_resign = float("-inf")
    loop_counter = 1
    training_counter = 0

    ## Running loop
    while True:
        print("Beginning loop", loop_counter)
        print("Beginning self-play")
        ## Generate new data for training
        with torch.no_grad():
            v_resign = sim_games(N_training_games,
                                 best_model,
                                 v_resign,
                                 MCTS_settings)

        writer.add_scalar('v_resign', v_resign, loop_counter)

        print("Begin training")
        ## Now train model
        training_model = trainer.train(training_model)

        print("Begin evaluation")
        ## Evaluate training model against best model
        # Below are the needed functions
        best_model.eval()
        training_model.eval()
        with torch.no_grad():
            scores = sim_games(N_duel_games,
                               training_model,
                               v_resign,
                               MCTS_settings,
                               model2=best_model,
                               duel=True)

        # Find best model
        # Here the model has to win atleast 60% of the games

        best_model = training_model
        save_model(best_model)

        new_elo, model_iter_counter = elo_league.common_duel_elo(scores[0] / (scores[1] + scores[0]))
        elo_league.save_league()
        print("The score was:")
        print(scores)
        print("New elo is: " + str(new_elo))
        # Store statistics
        writer.add_scalar('Elo', new_elo, model_iter_counter)
        loop_counter += 1
