import numpy as np
from collections import deque, Counter, defaultdict
from torch.multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
import time
import queue
import re
import os
import glob
from tqdm import tqdm
import sys
from storage_functions import experience_replay_sender
from MCTS import state_node, expand_node, backup_node, MCTS

# Import torch things
import torch


def gpu_worker(gpu_Q, MCTS_settings, model1, model2=None):
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        batch_test_length = 1000
        action_size = MCTS_settings["action_size"]
        obs_size = MCTS_settings["observation_size"]
        n_parallel_explorations = MCTS_settings["n_parallel_explorations"]
        cuda = torch.cuda.is_available()
        num_eval = 0
        pipe_queue = deque([])
        if cuda:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        # Calibrate queue size
        calibration_done = False
        MCTS_queue = 1
        t = time.time()
        # The +1 next line is to take cases where a small job is submitted
        batch = torch.empty((n_parallel_explorations * (MCTS_queue + 1),) + (obs_size,))
        speed = float("-inf")

        while True:
            # Loop to get data
            i = 0
            jobs_indexes = []
            queue_size = n_parallel_explorations * MCTS_queue
            while i < queue_size:
                try:
                    gpu_jobs, pipe = gpu_Q.get(True, 0.0001)  # Get jobs
                    pipe_queue.append(pipe)  # Track job index with pipe index
                    num_jobs = gpu_jobs.shape[0]  # Get number of jobs send
                    jobs_indexes.append(num_jobs)  # Track number of jobs send
                    batch[i:(i + num_jobs)] = torch.from_numpy(gpu_jobs)  # Add jobs to batch
                    num_eval += num_jobs  # Increase number of evaluations for speed check
                    i += num_jobs

                except queue.Empty:
                    if i != 0:
                        break
            # Evaluate and send jobs
            if model2 != None:
                # Case where worker uses f and g model
                f_g_process(batch, pipe_queue, jobs_indexes, model1, model2)
            else:
                # Case where worker uses h-model
                h_process(batch, pipe_queue, jobs_indexes, model1)

            # Update calibration
            if ((num_eval % (batch_test_length + 1)) == 0) and (not calibration_done):
                time_spent = time.time() - t
                t = time.time()
                new_speed = num_eval / time_spent
                calibration_done = new_speed < speed
                speed = new_speed

                f = open("speed.txt", "a")
                print("Queue size:" + str(MCTS_queue) + "Evals pr. second: " + str(round(speed)), file=f)
                f.close()

                MCTS_queue += 1 - 2 * calibration_done  # The calibration is for going back to optimal size
                batch = torch.empty((n_parallel_explorations * (MCTS_queue + 1),) + (obs_size,))
                num_eval = 0


def f_g_process(batch, pipe_queue, jobs_indexes, f_model, g_model):
    # Function for processing jobs for booth dynamic (g) and value (f) evaluation model
    # Process data
    S, u = g_model.forward(batch)
    result = f_model.forward(S)
    S = S.cpu().numpy()
    u = u.cpu().numpy()
    P = result[0].cpu().numpy()
    v = result[1].cpu().numpy()
    # Send processed jobs to all threads
    for jobs_index in jobs_indexes:
        index_end = index_start + jobs_index
        S_indexed = S[index_start:index_end]
        u_indexed = u[index_start:index_end]
        P_indexed = P[index_start:index_end]
        v_indexed = v[index_start:index_end]
        pipe_queue.popleft().send([S_indexed, u_indexed, P_indexed, v_indexed])
        index_start = index_end


def h_process(batch, pipe_queue, jobs_indexes, h_model):
    # Function for processing jobs for booth dynamic (g) and value (f) evaluation model
    # Process data
    S = h_model.forward(batch)
    S = S.cpu().numpy()
    # Send processed jobs to all threads
    for jobs_index in jobs_indexes:
        index_end = index_start + jobs_index
        S_indexed = S[index_start:index_end]
        pipe_queue.popleft().send([S_indexed])
        index_start = index_end


def sim_game(env, f_g_Q, h_Q, ex_Q, MCTS_settings, MuZero_settings, experience_settings):
    # Hyperparameters
    temp_switch = MuZero_settings["temp_switch"]  # Number of turns before other temperature measure is used
    eta_par = MuZero_settings["eta_par"]
    epsilon = MuZero_settings["epsilon"]
    action_size = MCTS_settings["action_size"]
    ER = experience_replay_sender(ex_Q, experience_settings)

    # Define pipe for f-, g-, h-model gpu workers
    f_g_rec, f_g_send = Pipe(False)
    h_rec, h_send = Pipe(False)

    # Start environment
    turns = 0

    # Generate root/first node
    root_node = state_node()
    # Start game
    S_obs = env.reset()
    h_Q.put([S_obs, h_send])  # Get hidden state of observation
    S = h_rec.recv()
    a = 0  # Dummy action
    stored_jobs = ([S, [], [], root_node, a])
    S_array, u_array, P_array, v_array = expand_node(stored_jobs, f_g_Q, f_g_send, f_g_rec, MCTS_settings)
    S = backup_node(stored_jobs, S_array, P_array, v_array, MCTS_settings)

    # Loop over all turns in environment
    while True:
        turns += 1
        root_node.set_illegal(env.illegal())
        if (turns <= temp_switch):
            # Case where early temperature is used
            # Simulate MCTS
            root_node = MCTS(root_node, f_g_Q, MCTS_settings)
            # Compute action distribution from policy
            pi_legal = root_node.N / root_node.N_total
            # Selecet action
            action = np.random.choice(action_size, size=1, p=pi_legal)[0]
        else:
            # Case where later temperature is used
            # Get noise
            eta = np.random.dirichlet(np.ones(action_size) * eta_par)
            root_node.P = (1 - epsilon) * root_node.P + epsilon * eta

            # Simulate MCTS
            root_node = MCTS(root_node, f_g_Q, MCTS_settings)

            # Compute legal actions visit count (needed for storing)
            pi_legal = root_node.N / root_node.N_total

            # Pick move
            action = np.argmax(root_node.N)
        Q = root_node.Q  # Store estimated values (G)
        # Pick move
        root_node = root_node.action_edges[action]
        S_new_obs, r, done, info = env.step(action)
        # Save Data
        ER.store([S_obs, S_new_obs, action, r, done, pi_legal, root_node.v], done)
        S_obs = S_new_obs

        h_Q.put([S_obs, h_send])
        S = h_rec.recv()
        if done:
            # Check for termination of environment
            break

    # Episode is over
    ER.send_episode()
