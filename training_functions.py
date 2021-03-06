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
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from storage_functions import experience_replay_server
from torch.utils.tensorboard import SummaryWriter
from models import muZero, calc_support_dist
from helper_functions import normal_support
import warnings
import copy

# 1. Disabled functions: action appending of hidden dynamic state
# 2. Exponential reduction of learning rate, instead reduce on platau
# 3. Unrolling of K
# 4. priority sampling
# 5. loss scaling
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


def squared_loss(u, r, z, v, pi, P, P_imp, N, beta, mean=True):
    # Loss used for the games go, chess, and shogi. This uses the 2-norm difference of values
    def l_r(u_tens, r_tens):
        assert(u_tens.shape == r_tens.shape)
        loss = (u_tens-r_tens)**2
        return loss

    def l_v(z_tens, v_tens):
        assert (z_tens.shape == v_tens.shape)
        loss = (z_tens-v_tens)**2
        return loss

    def l_p(pi_tens, P_tens):
        # Log is already applied to output P_tens
        assert (pi_tens.shape == P_tens.shape)
        loss = torch.sum(pi_tens*P_tens, dim=2)
        return loss

    reward_error = l_r(u, r)
    value_error = l_v(z, v)
    policy_error = -l_p(P, pi)
    total_error = reward_error + value_error + policy_error
    total_error = (total_error/(P_imp[:,None] * N))**beta  # Scale gradient with importance weighting
    if mean:
        return total_error.mean(), reward_error.mean(), value_error.mean(), policy_error.mean()
    else:
        return total_error, reward_error, value_error, policy_error

def bayes_muZero_loss(u, r, z, v, pi, P, P_imp, N, beta, n_iter=0, mean=True):
    # Loss used for atari. Assumes r, v, P are in logspace and sorftmaxed
    def cross_entropy(target, input):
        assert (target.shape == input.shape)
        loss = -torch.sum(target*input, dim=-1)
        return loss

    reward_error = cross_entropy(u, r)
    value_error = cross_entropy(z, v)
    policy_error = cross_entropy(pi, P)
    total_error = reward_error + value_error + policy_error.mean(-1)
    total_error = (total_error/(P_imp[:, None] * N))**beta  # Scale gradient with importance weighting
    if mean:
        return total_error.mean(), reward_error.mean(), value_error.mean(), policy_error.mean()
    else:
        return total_error, reward_error, value_error, policy_error


def muZero_games_loss(u, r, z, v, pi, P, P_imp, N, beta, n_iter=0, mean=True):
    # Loss used for atari. Assumes r, v, P are in logspace and sorftmaxed
    def cross_entropy(target, input):
        assert (target.shape == input.shape)
        loss = -torch.sum(target*input, dim=2)
        return loss

    reward_error = cross_entropy(u, r)
    value_error = cross_entropy(z, v)
    policy_error = cross_entropy(pi, P)
    total_error = reward_error + value_error + policy_error
    total_error = (total_error/(P_imp[:, None] * N))**beta  # Scale gradient with importance weighting
    if mean:
        return total_error.mean(), reward_error.mean(), value_error.mean(), policy_error.mean()
    else:
        return total_error, reward_error, value_error, policy_error


class model_trainer:
    def __init__(self, f_model, g_model, h_model, experience_replay, experience_settings, training_settings, MCTS_settings):
        self.MCTS_settings = MCTS_settings
        self.ER = experience_replay
        self.experience_settings = experience_settings
        self.K = experience_settings["K"]
        self.num_epochs = training_settings["num_epochs"]
        self.BS = training_settings["train_batch_size"]
        self.alpha = training_settings["alpha"]
        self.beta = training_settings["beta"]
        self.lr_init = training_settings["lr_init"]
        self.lr_decay_rate = training_settings["lr_decay_rate"]
        self.lr_decay_steps = training_settings["lr_decay_steps"]
        self.momentum = training_settings["momentum"]
        self.weight_decay = training_settings["weight_decay"]
        self.uniform_sampling = training_settings["uniform_sampling"]
        self.scale_values = training_settings["scale_values"]
        self.hidden_S_size = MCTS_settings["hidden_S_size"]
        self.action_size = MCTS_settings["action_size"]
        self.wr_Q = MCTS_settings["Q_writer"]
        self.bayesian = MCTS_settings["bayesian"]
        if training_settings["use_different_gpu"]:
            self.device = torch.device('cuda:' + str(torch.cuda.device_count()-1))
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.training_counter = 0

        # Check for cuda
        # GPU things
        self.cuda = torch.cuda.is_available()
        # Inference model
        self.f_model = f_model
        self.g_model = g_model
        self.h_model = h_model
        # Training model
        self.f_model_train = copy.deepcopy(self.f_model)
        self.g_model_train = copy.deepcopy(self.g_model)
        self.h_model_train = copy.deepcopy(self.h_model)
        self.f_model_train.to(self.device)
        self.g_model_train.to(self.device)
        self.h_model_train.to(self.device)

        self.muZero = muZero(self.f_model_train, self.g_model_train, self.h_model_train, self.K, self.hidden_S_size, self.action_size)
        #self.optimizer = optim.SGD(self.muZero.parameters(), lr=self.lr_init, momentum=self.momentum, weight_decay=self.weight_decay)
        self.optimizer = optim.Adam(self.muZero.parameters(), lr=self.lr_init, weight_decay=self.weight_decay)
        if MCTS_settings["bayesian"]:
            self.criterion = bayes_muZero_loss
        else:
            self.criterion = muZero_games_loss

        gamma = self.lr_decay_rate ** (1 / self.lr_decay_steps)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience=0, eps=1)

    def convert_torch(self, tensors):
        converted = []
        for tensor in tensors:
            converted.append(torch.from_numpy(tensor).to(self.device))
        return converted

    def train(self):
        self.f_model_train.train()
        self.g_model_train.train()
        self.h_model_train.train()

        length_training = self.num_epochs
        start_time = time.time()
        # Train
        for i in range(length_training):
            with torch.no_grad():
                # Generate batch
                S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, batch_idx, P_imp, N_count = self.ER.return_batches(self.BS,
                                                                                                                self.alpha,
                                                                                                                self.K,
                                                                                                                uniform_sampling=self.uniform_sampling)
                S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, P_imp = self.convert_torch([S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, P_imp])

                u_support_batch, u_scaled = calc_support_dist(u_batch.to(torch.float64), self.g_model_train.support, scale_value=self.scale_values)
                z_support_batch, z_scaled = calc_support_dist(z_batch.to(torch.float64), self.f_model_train.support, scale_value=self.scale_values)
                if self.bayesian:
                    pi_view = pi_batch[:, :, :, 0].view(-1, pi_batch.shape[2])
                    pi_view[range(pi_view.shape[0]), a_batch.view(-1)] = z_batch.view(-1)
                    pi_batch = normal_support(pi_batch[:, :, :, 0], pi_batch[:, :, :, 1], self.f_model_train.trans_support)
                #assert(torch.all(torch.isclose(pi_batch.sum(dim=2), torch.tensor(1.))))
                assert(torch.all(torch.isclose(u_support_batch.sum(dim=2), torch.tensor(1., dtype=torch.float64))))
                assert(torch.all(torch.isclose(z_support_batch.sum(dim=2), torch.tensor(1., dtype=torch.float64))))
                assert(torch.all(torch.isclose(self.g_model_train.dist2mean(u_support_batch.view(self.BS*self.K, -1), None), u_scaled.view(-1).to(torch.float64), atol=1e-06)))
                assert(torch.all(torch.isclose(self.f_model_train.dist2mean(z_support_batch.view(self.BS*self.K, -1), None), z_scaled.view(-1).to(torch.float64), atol=1e-06)))

            # Optimize
            self.optimizer.zero_grad()
            P_batches, v_batches, r_batches, p_vals = self.muZero.forward(S_batch, a_batch, z_batch)
            loss, r_loss, v_loss, P_loss = self.criterion(u_support_batch, r_batches,
                                                          z_support_batch, v_batches,
                                                          pi_batch, P_batches,
                                                          P_imp, N_count, self.beta, self.muZero.n_updates)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.muZero.n_updates += 1

            self.ER.update_weightings(p_vals.mean(axis=1), batch_idx)

            # Save model
            if self.muZero.n_updates % 1000 == 0:
                torch.save({'muzero_state_dict': self.muZero.state_dict(),
                            'f_model_state_dict': self.f_model_train.state_dict(),
                            'g_model_state_dict': self.g_model_train.state_dict(),
                            'h_model_state_dict': self.h_model_train.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           "checkpoints/MuZero_model_" + str(self.muZero.n_updates))

            # Log summary statistics
            if self.training_counter % 100 == 1:
                # Log summary statistics
                self.wr_Q.put(['dist', 'Output/v', v_batches.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Output/P', P_batches.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Output/r', r_batches.detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Output/P_max', torch.mean(torch.max(P_batches, dim=2)[0]).detach().cpu(),
                               self.training_counter])
                r_trans = self.g_model_train.dist2mean(r_batches.view(self.BS*self.K, -1).exp(), self.scale_values)
                v_trans = self.f_model_train.dist2mean(v_batches.view(self.BS*self.K, -1).exp(), self.scale_values)
                self.wr_Q.put(['dist', 'Output/transformed_r', r_trans.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Output/transformed_v', v_trans.detach().cpu(), self.training_counter])
                # Check for biases of actions
                self.wr_Q.put(['scalar', 'stats/action0_pref', P_batches[:, :, 0].exp().mean().detach().cpu(),
                               self.training_counter])
                self.wr_Q.put(['scalar', 'stats/action0_play', (a_batch == 0).to(torch.float).mean().detach().cpu(),
                               self.training_counter])

                u_trans = self.g_model_train.dist2mean(u_support_batch.view(self.BS * self.K, -1), self.scale_values)
                z_trans = self.f_model_train.dist2mean(z_support_batch.view(self.BS * self.K, -1), self.scale_values)
                self.wr_Q.put(['dist', 'data/u_retransformed',  u_trans.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'data/z_retransformed', z_trans.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'data/u', u_batch.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'data/z', z_batch.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'data/Pi', pi_batch.detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'data/Pi_max', torch.mean(torch.max(pi_batch, dim=2)[0]).detach().cpu(),
                               self.training_counter])

                self.wr_Q.put(['scalar', 'Total_loss/train', loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Reward_loss/train', r_loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Value_loss/train', v_loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Policy_loss/train', P_loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Policy_loss_dist/train', P_loss.detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'learning_rate', self.scheduler._last_lr[0], self.training_counter])
                self.wr_Q.put(['scalar', 'replay/priority_mean', (N_count*P_imp).mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(
                    ['dist', 'replay/priority_dist', (N_count * P_imp).detach().cpu(), self.training_counter])

            self.training_counter += 1
        # Update network weights
        self.f_model.load_state_dict(self.f_model_train.state_dict())
        self.g_model.load_state_dict(self.g_model_train.state_dict())
        self.h_model.load_state_dict(self.h_model_train.state_dict())
        time.sleep(0.01)
        # Also time iteration
        end_time = time.time()
        speed = length_training / (end_time - start_time)
        self.wr_Q.put(['scalar', 'Other/iterations_pr_sec', speed, self.training_counter])


def train_ex_worker(ex_Q, f_model, g_model, h_model, experience_settings, training_settings, MCTS_settings):
    ER_server = experience_replay_server(ex_Q, experience_settings, MCTS_settings)
    trainer = model_trainer(f_model, g_model, h_model, ER_server, experience_settings, training_settings, MCTS_settings)
    while True:
        # Empty queue
        while (ER_server.total_store < training_settings["train_batch_size"]) or not ex_Q.empty():
            ER_server.recv_store()
        # Train
        trainer.train()


class fix_out_of_order:
    # Because Tensorboard for some reason connects datapoints based on time added and not x-axis values,
    #  this function sorts them....
    def __init__(self, name_list, writer):
        self.name_list = name_list
        self.writer = writer
        self.buffers = [[] for i in range(len(name_list))]
        self.name2idx = {}
        self.current_idx = np.ones((len(name_list),))
        self.dicts = [{} for i in range(len(name_list))]
        for i in range(len(name_list)):
            self.name2idx[name_list[i]] = i

    def update(self, type, name, value, index):
        if name in self.name_list:
            name_idx = self.name2idx[name]
            self.dicts[name_idx][index] = [type, name, value, index]

            while self.current_idx[name_idx] in self.dicts[name_idx]:
                # Add next number if exists
                type, name, value, index = self.dicts[name_idx].pop(self.current_idx[name_idx])
                write_point(self.writer, type, name, value, index)
                self.current_idx[name_idx] += 1
        else:
            write_point(self.writer, type, name, value, index)


def write_point(writer, type, name, value, index):
    if type == 'scalar':
        writer.add_scalar(name, value, index)
    elif type == 'dist':
        writer.add_histogram(name, value, index)
    else:
        print("Writer error")
        raise TypeError


def writer_worker(wr_Q):
    np.random.seed(0)
    writer = SummaryWriter()
    name_list = ["environment/steps", "environment/total_reward", "environment/iter_pr_sec"]
    updater = fix_out_of_order(name_list, writer)
    while True:
        # Empty queue
        while not wr_Q.empty():
            type, name, value, index = wr_Q.get()
            updater.update(type, name, value, index)

        time.sleep(1)