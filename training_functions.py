#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:40:57 2019

@author: tirsgaard
"""
import re
import os
import glob
import numpy as np
import torch
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import warnings

def save_model(model):
    subdirectory = "model/saved_models/"
    os.makedirs(subdirectory, exist_ok=True)
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory + "*.model"):
        files.append(file)

    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "", file)))

    if (number_list == []):
        number_list = [0]
    # Get max number
    latest_new_model = max(number_list) + 1

    save_name = subdirectory + "model_" + str(latest_new_model) + ".model"
    torch.save(model, save_name)


def load_latest_model():
    subdirectory = "model/saved_models/"
    os.makedirs(subdirectory, exist_ok=True)
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory + "*.model"):
        files.append(file)

    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "", file)))

    if (number_list == []):
        warnings.warn("No model was found in path " + subdirectory, RuntimeWarning)
        return None
    # Get max number
    latest_model = max(number_list)

    load_name = subdirectory + "model_" + str(latest_model) + ".model"
    print("Loading model " + load_name)
    sys.path.append("model")
    if (torch.cuda.is_available()):
        print("Using GPU")
        model = torch.load(load_name)
    else:
        print("Using CPU")
        model = torch.load(load_name, map_location=torch.device('cpu'))
    return model


def muZero_games_loss(u, r, z, v, pi, P, P_imp, N, beta):
    # Loss used for the games go, chess, and shogi. This uses the 2-norm difference of values
    def l_r(u_tens, r_tens):
        loss = (u_tens-r_tens)**2
        return loss

    def l_v(z_tens, v_tens):
        loss = (z_tens-v_tens)**2
        return loss

    def l_p(pi_tens, P_tens):
        loss = torch.sum((pi_tens-P_tens)**2, dim=2)
        return loss

    reward_error = l_r(u, r)
    value_error = l_v(z, v)
    policy_error = l_p(pi, P)
    total_error = reward_error + value_error + policy_error
    total_error = torch.mean((total_error/(P_imp[:,None] * N))**beta)  # Scale gradient with importance weighting
    return total_error, reward_error.mean(), value_error.mean(), policy_error.mean()


class model_trainer:
    def __init__(self, writer, expereince_replay, experience_settings, training_settings, MCTS_settings):
        self.writer = writer
        self.MCTS_settings = MCTS_settings
        self.criterion = muZero_games_loss
        self.ER = expereince_replay
        self.K = experience_settings["K"]
        self.num_epochs = training_settings["num_epochs"]
        self.bs = training_settings["train_batch_size"]
        self.alpha = training_settings["alpha"]
        self.beta = training_settings["beta"]
        self.lr_init = training_settings["lr_init"]
        self.lr_decay_rate = training_settings["lr_decay_rate"]
        self.lr_decay_steps = training_settings["lr_decay_steps"]
        self.momentum = training_settings["momentum"]
        self.training_counter = 0

        # Check for cuda
        # GPU things
        self.cuda = torch.cuda.is_available()

    def convert_torch(self, tensors):
        converted = []
        for tensor in tensors:
            converted.append(torch.from_numpy(tensor).float())
        # Convert to cuda if GPU support
        if self.cuda:
            for element in converted:
                element.cuda()

        return converted

    def train(self, f_model, g_model, h_model):
        f_model.train()
        g_model.train()
        h_model.train()
        model_list = list(f_model.parameters()) + list(h_model.parameters()) + list(g_model.parameters())
        optimizer = optim.SGD(model_list, lr=self.lr_init)
        gamma = self.lr_decay_rate ** (1 / self.lr_decay_steps)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        length_training = self.num_epochs
        # Train
        for i in range(length_training):
            # Generate batch. Note we uniform sample instead of epoch as in the original paper
            S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, batch_idx, P_imp = self.ER.return_batches(self.bs,
                                                                                                            self.alpha,
                                                                                                            self.K)
            S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, P_imp = self.convert_torch([S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, P_imp])
            z_batch = u_batch.clone()

            assert(torch.any(z_batch[:, 0] == S_batch[:, -1, 0, 0]))
            assert(torch.any(u_batch[:, 0] == S_batch[:, -1, 0, 0]))
            assert(torch.any(z_batch[:, 0] + 1 == z_batch[:, 1]))
            assert(torch.any(u_batch[:, 0] + 1 == u_batch[:, 1]))
            p_vals = []  # Number
            total_loss = []
            r_batches = []
            v_batches = []
            P_batches = []
            # Optimize
            optimizer.zero_grad()
            new_S = h_model.forward(S_batch)
            for k in range(self.K):
                P_batch, v_batch = f_model.forward(new_S)
                new_S, r_batch = g_model.forward(new_S)

                p_vals.append(torch.abs(v_batch.squeeze(dim=1) - z_batch[:,k]).detach().numpy())  # For importance weighting
                P_batches.append(P_batch)
                v_batches.append(v_batch)
                r_batches.append(r_batch)

            P_batches = torch.stack(P_batches, dim=1)
            v_batches = torch.stack(v_batches, dim=1).squeeze(dim=2)
            r_batches = torch.stack(r_batches, dim=1).squeeze(dim=2)
            self.ER.update_weightings(p_vals[0], batch_idx)
            loss, r_loss, v_loss, P_loss = self.criterion(u_batch, r_batches.squeeze(dim=1),
                                                          z_batch, v_batches.squeeze(dim=1),
                                                          pi_batch, P_batches,
                                                          P_imp, self.ER.N, self.beta)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if self.training_counter % 50 == 1:
                try:
                    #self.writer.flush()
                    self.writer.add_histogram('Output/v', v_batch, self.training_counter)
                    self.writer.add_histogram('Output/P', P_batch, self.training_counter)
                    self.writer.add_histogram('Output/r', r_batch, self.training_counter)
                    self.writer.add_histogram('Output/Pi', pi_batch, self.training_counter)

                    self.writer.add_histogram('data/u', u_batch, self.training_counter)
                    self.writer.add_histogram('data/z', z_batch, self.training_counter)
                    self.writer.add_histogram('data/Pi', pi_batch, self.training_counter)

                    self.writer.add_scalar('Total_loss/train', loss.detach(), self.training_counter)
                    self.writer.add_scalar('Reward_loss/train', r_loss.detach(), self.training_counter)
                    self.writer.add_scalar('Value_loss/train', v_loss.detach(), self.training_counter)
                    self.writer.add_scalar('Policy_loss/train', P_loss.detach(), self.training_counter)
                    self.writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], self.training_counter)
                except:
                    return
            self.training_counter += 1