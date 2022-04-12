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
import time
import torch
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from storage_functions import experience_replay_server
from torch.utils.tensorboard import SummaryWriter
from models import stack_a_torch
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
    def __init__(self, f_model, g_model, h_model, expereince_replay, experience_settings, training_settings, MCTS_settings):
        self.MCTS_settings = MCTS_settings
        self.criterion = muZero_games_loss
        self.ER = expereince_replay
        self.K = experience_settings["K"]
        self.num_epochs = training_settings["num_epochs"]
        self.BS = training_settings["train_batch_size"]
        self.alpha = training_settings["alpha"]
        self.beta = training_settings["beta"]
        self.lr_init = training_settings["lr_init"]
        self.lr_decay_rate = training_settings["lr_decay_rate"]
        self.lr_decay_steps = training_settings["lr_decay_steps"]
        self.momentum = training_settings["momentum"]
        self.hidden_S_size = MCTS_settings["hidden_S_size"]
        self.action_size = MCTS_settings["action_size"]
        self.wr_Q = MCTS_settings["Q_writer"]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.training_counter = 0

        # Check for cuda
        # GPU things
        self.cuda = torch.cuda.is_available()
        # Learner
        self.f_model = f_model
        self.g_model = g_model
        self.h_model = h_model
        model_list = list(self.f_model.parameters()) + list(self.h_model.parameters()) + list(self.g_model.parameters())
        self.optimizer = optim.SGD(model_list, lr=self.lr_init)
        gamma = self.lr_decay_rate ** (1 / self.lr_decay_steps)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)

    def convert_torch(self, tensors):
        converted = []
        for tensor in tensors:
            converted.append(torch.from_numpy(tensor).to(self.device))
        return converted

    def train(self):
        self.f_model.to(self.device).train()
        self.g_model.to(self.device).train()
        self.h_model.to(self.device).train()


        length_training = self.num_epochs
        # Train
        for i in range(length_training):
            # Generate batch. Note we uniform sample instead of epoch as in the original paper
            S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, batch_idx, P_imp = self.ER.return_batches(self.BS,
                                                                                                            self.alpha,
                                                                                                            self.K)
            S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, P_imp = self.convert_torch([S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, P_imp])
            p_vals = []  # Number
            r_batches = []
            v_batches = []
            P_batches = []
            # Optimize
            self.optimizer.zero_grad()
            new_S2 = self.h_model.forward(S_batch)
            new_S = new_S2
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
            self.ER.update_weightings(p_vals[0], batch_idx)
            loss, r_loss, v_loss, P_loss = self.criterion(u_batch, r_batches.squeeze(dim=1),
                                                          z_batch, v_batches.squeeze(dim=1),
                                                          pi_batch, P_batches,
                                                          P_imp, self.ER.N, self.beta)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.training_counter % 10 == 1:
                self.wr_Q.put(['dist', 'Output/v', v_batch.detach().cpu().numpy(), self.training_counter])
                self.wr_Q.put(['dist', 'Output/P', P_batch.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Output/r', r_batch.detach().cpu(), self.training_counter])

                self.wr_Q.put(['dist', 'data/u', u_batch.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'data/z', z_batch.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'data/Pi', pi_batch.detach().cpu(), self.training_counter])

                self.wr_Q.put(['scalar', 'Total_loss/train', loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Reward_loss/train', r_loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Value_loss/train', v_loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Policy_loss/train', P_loss.detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'learning_rate', self.scheduler.get_last_lr()[0], self.training_counter])

            if self.training_counter % 10000 == 0:
                # Weights
                self.wr_Q.put(['dist', 'training/model_f/layer0', list(self.f_model.parameters())[0].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_f/layer6', list(self.f_model.parameters())[6].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_f/layer17', list(self.f_model.parameters())[17].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_f/layer18', list(self.f_model.parameters())[18].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_f/layer22', list(self.f_model.parameters())[22].detach().cpu(), self.training_counter])

                self.wr_Q.put(['dist', 'training/model_h/layer0', list(self.h_model.parameters())[0].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_h/layer3', list(self.h_model.parameters())[3].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_h/layer9', list(self.h_model.parameters())[9].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_h/layer18', list(self.h_model.parameters())[18].detach().cpu(), self.training_counter])

                self.wr_Q.put(['dist', 'training/model_g/layer0', list(self.g_model.parameters())[0].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_g/layer3', list(self.g_model.parameters())[3].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_g/layer9', list(self.g_model.parameters())[9].detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'training/model_g/layer18', list(self.g_model.parameters())[18].detach().cpu(), self.training_counter])

                # Gradients
                self.wr_Q.put(['dist', 'gradient/model_f/layer0', list(self.f_model.parameters())[0].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_f/layer6', list(self.f_model.parameters())[6].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_f/layer17', list(self.f_model.parameters())[17].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_f/layer18', list(self.f_model.parameters())[18].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_f/layer22', list(self.f_model.parameters())[22].grad.detach().cpu(), self.training_counter])

                self.wr_Q.put(['dist', 'gradient/model_h/layer0', list(self.h_model.parameters())[0].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_h/layer3', list(self.h_model.parameters())[3].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_h/layer9', list(self.h_model.parameters())[9].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_h/layer18', list(self.h_model.parameters())[18].grad.detach().cpu(), self.training_counter])

                self.wr_Q.put(['dist', 'gradient/model_g/layer0', list(self.g_model.parameters())[0].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_g/layer3', list(self.g_model.parameters())[3].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_g/layer9', list(self.g_model.parameters())[9].grad.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'gradient/model_g/layer18', list(self.g_model.parameters())[18].grad.detach().cpu(), self.training_counter])


            self.training_counter += 1



def train_ex_worker(ex_Q, f_model, g_model, h_model, experience_settings, training_settings, MCTS_settings):
    ER_server = experience_replay_server(ex_Q, experience_settings, MCTS_settings)
    trainer = model_trainer(f_model, g_model, h_model, ER_server, experience_settings, training_settings, MCTS_settings)
    while True:
        # Empty queue
        while (ER_server.total_store < training_settings["train_batch_size"]) or not ex_Q.empty():
            ER_server.recv_store()
        # Train
        trainer.train()

def writer_worker(wr_Q):
    writer = SummaryWriter()
    while True:
        # Empty queue
        while not wr_Q.empty():
            type, name, value, index = wr_Q.get()
            if type == 'scalar':
                writer.add_scalar(name, value, index)
            elif type == 'dist':
                writer.add_histogram(name, value, index)
            else:
                raise TypeError

        time.sleep(1)