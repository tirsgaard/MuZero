#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:51:02 2019

@author: tirsgaard
"""

from game_functions import sim_games
from training_functions import save_model, load_latest_model, model_trainer
from storage_functions import experience_replay_server
from models import dummy_networkF, dummy_networkG, dummy_networkH
from torch.multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
import gym
from wrappers import MaxAndSkipEnv, RAMAndSkipEnv, RAM_Breakout
import hyperparameters as conf
from game_functions import sim_game, gpu_worker
from go_model import ConvResNet, ResNet, ResNet_f, ResNet_g
from models import ram_network_convG, ram_network_convH, ram_network_convF
import numpy as np


import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # Get configs
    MuZero_settings = conf.MuZero_settings
    experience_settings = conf.experience_settings
    MCTS_settings = conf.MCTS_settings
    training_settings = conf.training_settings
    MuZero_settings["N_training_games"] = 64
    # Construct networks
    observation_shape = (experience_settings["past_obs"]*MCTS_settings["observation_channels"],) + MCTS_settings["observation_size"]  # input to f
    hidden_shape = (MCTS_settings["hidden_S_channel"], ) + MCTS_settings["hidden_S_size"]  # input to f
    action_size = MCTS_settings["action_size"]
    hidden_input_size = (MCTS_settings["action_size"][0] + MCTS_settings["hidden_S_channel"],) + MCTS_settings["hidden_S_size"]  # Input to g
    n_heads = MuZero_settings["n_support"]
    transform_values = training_settings["scale_values"]
    torch.manual_seed(0)
    np.random.seed(1)
    support = torch.linspace(MuZero_settings["low_support"], MuZero_settings["high_support"], n_heads)

    model_name = "checkpoints/MuZero_model_8000"

    f_model = ram_network_convF(hidden_shape, MCTS_settings["action_size"], 1024, support, transform_values)
    g_model = ram_network_convG(hidden_input_size, hidden_shape, 1024, support, transform_values)
    h_model = ram_network_convH(observation_shape, hidden_shape, 128)
    checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
    f_model.load_state_dict(checkpoint['f_model_state_dict'])
    g_model.load_state_dict(checkpoint['g_model_state_dict'])
    h_model.load_state_dict(checkpoint['h_model_state_dict'])

    torch.multiprocessing.set_start_method('spawn', force=True)
    Q_writer = Queue()
    training_settings["Q_writer"] = Q_writer
    experience_settings["Q_writer"] = Q_writer
    MCTS_settings["Q_writer"] = Q_writer
    # Construct model trainer and experience storage
    EX_Q = Queue()
    ER = experience_replay_server(EX_Q, experience_settings, MCTS_settings)
    trainer = model_trainer(f_model, g_model, h_model, ER, experience_settings, training_settings, MCTS_settings)
    env_maker = lambda: RAMAndSkipEnv(gym.make("ALE/Breakout-v5", full_action_space=False, obs_type="ram"))

    # Make queues for sending data
    f_g_Q = Queue()
    h_f_Q = Queue()

    process_workers = []
    S_size = observation_shape#(experience_settings["past_obs"], ) + MCTS_settings["observation_size"]
    process_workers.append(
        Process(target=gpu_worker, args=(f_g_Q, hidden_input_size, MCTS_settings, g_model, f_model, True)))
    process_workers.append(
        Process(target=gpu_worker, args=(h_f_Q, S_size, MCTS_settings, h_model, f_model, False)))
    # Also make pipe for receiving v_resign
    conn_rec, conn_send = Pipe(False)

    with torch.no_grad():
        # Make process for gpu worker and data_handler

        # Start gpu and data_loader worker
        for p in process_workers:
            p.start()
        #sim_games(RAM_Breakout, f_model, g_model, h_model, EX_Q, MCTS_settings, MuZero_settings, experience_settings)
        for i in range(100):
            sim_game(RAM_Breakout, 0, 0, f_g_Q, h_f_Q, EX_Q, MCTS_settings, MuZero_settings, experience_settings)
    # Close processes
    for p in process_workers:
        p.terminate()
        p.join()
        EX_Q.cancel_join_thread()

    import psutil
    import os
    import sys

    def kill_child_proc(ppid):
        for process in psutil.process_iter():
            _ppid = process.ppid()
            if _ppid == ppid:
                _pid = process.pid
                if sys.platform == 'win32':
                    process.terminate()
                else:
                    os.system('kill -9 {0}'.format(_pid))


    pid = os.getpid()
    kill_child_proc(pid)