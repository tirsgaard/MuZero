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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from storage_functions import experience_replay_server
from torch.utils.tensorboard import SummaryWriter
from models import stack_a_torch
import torch.nn as nn
from models import muZero
import warnings
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
        assert (pi_tens.shape == P_tens.shape)
        loss = torch.sum((pi_tens-P_tens)**2, dim=2)
        return loss

    reward_error = l_r(u, r)
    value_error = l_v(z, v)
    policy_error = l_p(pi, P)
    total_error = reward_error + value_error + policy_error
    total_error = (total_error/(P_imp[:,None] * N))**beta  # Scale gradient with importance weighting
    #total_error = torch.mean((total_error / N) ** beta)  # Scale gradient without importance weighting
    if mean:
        return total_error.mean(), reward_error.mean(), value_error.mean(), policy_error.mean()
    else:
        return total_error, reward_error, value_error, policy_error


def muZero_games_loss(u, r, z, v, pi, P, P_imp, N, beta, mean=True):
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
        assert (pi_tens.shape == P_tens.shape)
        loss = torch.sum(pi_tens*torch.log(P_tens), dim=2)
        return loss

    reward_error = l_r(u, r)
    value_error = l_v(z, v)
    policy_error = l_p(pi, P)
    total_error = reward_error + value_error + policy_error
    total_error = torch.mean((total_error/(P_imp[:,None] * N))**beta)  # Scale gradient with importance weighting
    #total_error = torch.mean((total_error / N) ** beta)  # Scale gradient without importance weighting
    if mean:
        return total_error.mean(), reward_error.mean(), value_error.mean(), policy_error.mean()
    else:
        return total_error, reward_error, value_error, policy_error


class model_trainer:
    def __init__(self, f_model, g_model, h_model, experience_replay, experience_settings, training_settings, MCTS_settings):
        self.MCTS_settings = MCTS_settings
        self.criterion = squared_loss
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
        self.muZero = muZero(self.f_model, self.g_model, self.h_model, self.K, self.hidden_S_size, self.action_size)
        #self.optimizer = optim.SGD(self.muZero.parameters(), lr=self.lr_init, momentum=self.momentum, weight_decay=self.weight_decay)
        self.optimizer = optim.Adam(self.muZero.parameters(), lr=self.lr_init, weight_decay=self.weight_decay)

        gamma = self.lr_decay_rate ** (1 / self.lr_decay_steps)
        #self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience=0, eps=1)

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
        start_time = time.time()
        # Train
        for i in range(length_training):
            # Generate batch. Note we uniform sample instead of epoch as in the original paper
            S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, batch_idx, P_imp, N_count = self.ER.return_batches(self.BS,
                                                                                                            self.alpha,
                                                                                                            self.K,
                                                                                                            uniform_sampling=self.uniform_sampling)
            S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, P_imp = self.convert_torch([S_batch, a_batch, u_batch, done_batch, pi_batch, z_batch, P_imp])

            # Optimize
            self.optimizer.zero_grad()
            P_batches, v_batches, r_batches, p_vals = self.muZero.forward(S_batch, a_batch, z_batch)
            loss, r_loss, v_loss, P_loss = self.criterion(u_batch, r_batches,
                                                          z_batch, v_batches,
                                                          pi_batch, P_batches,
                                                          P_imp, N_count, self.beta)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

            self.ER.update_weightings(p_vals.mean(axis=1), batch_idx)

            # Log summary statistics
            if self.training_counter % 100 == 1:
                """
                z_loss = 0
                n_boot = self.experience_settings["n_bootstrap"]
                gamma = self.MCTS_settings["gamma"]
                for j in range(self.BS):
                    done_mask = done_batch[j].detach().cpu()
                    done_mask[np.argmax(done_mask)] = 0  # Set the first encountered done value to be 0
                    for k in range(self.K-n_boot):  # This is the interval done values can be reasoned
                        # Prepare done tiling
                        mask = torch.ones(n_boot)  # +1 to take bootstrap value into account
                        mask[:(self.K-k)] = done_mask[k:(k + n_boot)]
                        mask = torch.abs(1-mask)  # Invert mask
                        v_vals = torch.arange(u_batch[j, k], u_batch[j, k]+n_boot)
                        v_vals = v_vals * mask  # Remove done values
                        v_vals = torch.sum(v_vals*(gamma**torch.arange(0, n_boot)))
                        v_vals += mask[-1] * (S_batch[j, -1, 0, 0, 0] + k + n_boot-1) * gamma**n_boot
                        z_loss = max(torch.abs(v_vals-z_batch[j, k]), z_loss)

                self.wr_Q.put(['scalar', 'oracle/max_v_loss', z_loss.detach().cpu(),
                               self.training_counter])
                
                self.wr_Q.put(['scalar', 'oracle/r', torch.max(torch.abs(S_batch[:, -1, 0, 0, 0] - u_batch[:, 0])).detach().cpu(), self.training_counter])
                for k in range(1, self.K):
                    self.wr_Q.put(['scalar', 'oracle/r' + str(k+1), torch.max(torch.abs(S_batch[:, -1, 0, 0, 0] + k - u_batch[:, k])).detach().cpu(),
                     self.training_counter])
                """
                # Log summary statistics
                self.wr_Q.put(['dist', 'Output/v', v_batches.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Output/P', P_batches.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Output/r', r_batches.detach().cpu(), self.training_counter])

                self.wr_Q.put(['dist', 'data/u', u_batch.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'data/z', z_batch.detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'data/Pi', pi_batch.detach().cpu(), self.training_counter])

                self.wr_Q.put(['scalar', 'Total_loss/train', loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Reward_loss/train', r_loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Value_loss/train', v_loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'Policy_loss/train', P_loss.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(['dist', 'Policy_loss_dist/train', P_loss.detach().cpu(), self.training_counter])
                self.wr_Q.put(['scalar', 'learning_rate', self.scheduler._last_lr[0], self.training_counter])
                self.wr_Q.put(['scalar', 'replay/priority_mean', N_count*P_imp.mean().detach().cpu(), self.training_counter])
                self.wr_Q.put(
                    ['dist', 'replay/priority_dist', (N_count * P_imp).detach().cpu(), self.training_counter])


                # Check accuracy change over unrolls
                loss, r_loss, v_loss, P_loss = self.criterion(u_batch, r_batches,
                                                              z_batch, v_batches,
                                                              pi_batch, P_batches,
                                                              P_imp, self.ER.N, self.beta,  mean=False)
                loss_median, _ = loss.median(dim=0)
                r_loss_median, _ = r_loss.median(dim=0)
                v_loss_median, _ = v_loss.median(dim=0)
                P_loss_median, _ = P_loss.median(dim=0)
                for k in range(1, self.K):
                    self.wr_Q.put(['scalar', 'unrolling_errors/loss' + str(k + 1),
                                   (loss_median[k]/loss_median[0]).detach().cpu(),
                                   self.training_counter])
                    self.wr_Q.put(['scalar', 'unrolling_errors/r_loss' + str(k + 1),
                                   (r_loss_median[k] / r_loss_median[0]).detach().cpu(),
                                   self.training_counter])
                    self.wr_Q.put(['scalar', 'unrolling_errors/v_loss' + str(k + 1),
                                   (v_loss_median[k] / v_loss_median[0]).detach().cpu(),
                                   self.training_counter])
                    self.wr_Q.put(['scalar', 'unrolling_errors/P_loss' + str(k + 1),
                                   (P_loss_median[k] / P_loss_median[0]).detach().cpu(),
                                   self.training_counter])
                # Log gradients
                # Weights
                i = 0
                gradients = []
                for parms in list(self.f_model.parameters()):
                    self.wr_Q.put(['dist', 'training/model_f/layer' + str(i), parms.detach().cpu(),
                                   self.training_counter])
                    self.wr_Q.put(['dist', 'gradient/model_f/layer' + str(i), torch.abs(parms.grad).detach().cpu(),
                                   self.training_counter])
                    self.wr_Q.put(['scalar', 'mean_gradient/model_f/layer' + str(i), torch.abs(parms.grad).mean().detach().cpu(),
                                   self.training_counter])
                    gradients.append(torch.abs(parms.grad).mean().detach().cpu())
                    i += 1
                mean_grad = torch.median(torch.tensor(gradients))
                self.wr_Q.put(
                    ['scalar', 'median_gradient/model_f', mean_grad.detach().cpu(), self.training_counter])

                i = 0
                gradients = []
                for parms in list(self.g_model.parameters()):
                    self.wr_Q.put(
                        ['dist', 'training/model_g/layer' + str(i), parms.detach().cpu(),
                         self.training_counter])
                    self.wr_Q.put(['dist', 'gradient/model_g/layer' + str(i), torch.abs(parms.grad).detach().cpu(),
                                   self.training_counter])
                    self.wr_Q.put(
                        ['scalar', 'mean_gradient/model_g/layer' + str(i), torch.abs(parms.grad).mean().detach().cpu(),
                         self.training_counter])
                    gradients.append(torch.abs(parms.grad).mean().detach().cpu())
                    i += 1
                mean_grad = torch.median(torch.tensor(gradients))
                self.wr_Q.put(['scalar', 'median_gradient/model_g', mean_grad.detach().cpu(), self.training_counter])

                i = 0
                gradients = []
                for parms in list(self.h_model.parameters()):
                    self.wr_Q.put(
                        ['dist', 'training/model_h/layer' + str(i), parms.detach().cpu(),
                         self.training_counter])
                    self.wr_Q.put(['dist', 'gradient/model_h/layer' + str(i), torch.abs(parms.grad).detach().cpu(),
                                   self.training_counter])
                    self.wr_Q.put(
                        ['scalar', 'mean_gradient/model_h/layer' + str(i), torch.abs(parms.grad).mean().detach().cpu(),
                         self.training_counter])
                    gradients.append(torch.abs(parms.grad).mean().detach().cpu())
                    i += 1
                mean_grad = torch.median(torch.tensor(gradients))
                self.wr_Q.put(['scalar', 'median_gradient/model_h', mean_grad.detach().cpu(), self.training_counter])

            self.training_counter += 1
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
    name_list = ["environment/steps", "environment/total_reward"]
    updater = fix_out_of_order(name_list, writer)
    while True:
        # Empty queue
        while not wr_Q.empty():
            type, name, value, index = wr_Q.get()
            updater.update(type, name, value, index)

        time.sleep(1)