import numpy as np
from collections import deque, Counter, defaultdict
from torch.multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
import time
import queue
from storage_functions import experience_replay_sender
from MCTS import state_node, expand_node, backup_node, MCTS, generate_root
from tqdm import tqdm

# Import torch things
import torch


def gpu_worker(gpu_Q, input_shape, MCTS_settings, model, f_model, use_g_model):
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        batch_test_length = 1000
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
        batch = torch.empty((n_parallel_explorations * (MCTS_queue + 1),) + input_shape)
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
            if use_g_model:
                # Case where worker uses f and g model
                f_g_process(batch, pipe_queue, jobs_indexes, model, f_model)
            else:
                # Case where worker uses f and h model
                h_f_process(batch, pipe_queue, jobs_indexes, model, f_model)

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
                batch = torch.empty((n_parallel_explorations * (MCTS_queue + 1),) + input_shape)
                num_eval = 0


def f_g_process(batch, pipe_queue, jobs_indexes, g_model, f_model):
    # Function for processing jobs for booth dynamic (g) and value (f) evaluation model
    # Process data
    S, u = g_model.forward(batch)
    result = f_model.forward(S)
    S = S.cpu().numpy()
    u = u.cpu().numpy()
    P = result[0].cpu().numpy()
    v = result[1].cpu().numpy()
    index_start = 0
    # Send processed jobs to all threads
    for jobs_index in jobs_indexes:
        index_end = index_start + jobs_index
        S_indexed = S[index_start:index_end]
        u_indexed = u[index_start:index_end]
        P_indexed = P[index_start:index_end]
        v_indexed = v[index_start:index_end]
        pipe_queue.popleft().send([S_indexed, u_indexed, P_indexed, v_indexed])
        index_start = index_end


def h_f_process(batch, pipe_queue, jobs_indexes, h_model, f_model):
    # Function for processing jobs for booth dynamic (g) and value (f) evaluation model
    # Process data
    S = h_model.forward(batch)
    result = f_model.forward(S)
    S = S.cpu().numpy()
    P = result[0].cpu().numpy()
    v = result[1].cpu().numpy()
    index_start = 0
    # Send processed jobs to all threads
    for jobs_index in jobs_indexes:
        index_end = index_start + jobs_index
        S_indexed = S[index_start:index_end]
        P_indexed = P[index_start:index_end]
        v_indexed = v[index_start:index_end]
        pipe_queue.popleft().send([S_indexed, P_indexed, v_indexed])
        index_start = index_end
def temperature_scale(N, temp):
    N_temp = N**temp
    return N_temp/np.sum(N_temp)

def sim_game(env_maker, game_id, agent_id, f_g_Q, h_Q, EX_Q, MCTS_settings, MuZero_settings, experience_settings):
    # Hyperparameters
    temp_switch = MuZero_settings["temp_switch"]  # Number of turns before other temperature measure is used
    eta_par = MuZero_settings["eta_par"]
    epsilon = MuZero_settings["epsilon"]
    action_size = MCTS_settings["action_size"]
    wr_Q = MCTS_settings["Q_writer"]
    n_actions = np.prod(action_size)
    ER = experience_replay_sender(EX_Q, agent_id, MCTS_settings["gamma"], experience_settings)

    # Define pipe for f-, g-, h-model gpu workers
    f_g_rec, f_g_send = Pipe(False)
    h_rec, h_send = Pipe(False)

    # Make environment
    env = env_maker()
    total_R = 0
    turns = 0
    # Start game
    S_obs = env.reset()
    # Loop over all turns in environment
    while True:
        turns += 1
        #env.render()
        # Generate new tree, to throw away old values
        root_node = generate_root(S_obs, h_Q, f_g_Q, h_send, h_rec, f_g_send, f_g_rec, MCTS_settings)
        #root_node.set_illegal(env.illegal())
        # Simulate MCTS
        root_node = MCTS(root_node, f_g_Q, MCTS_settings)
        # Compute action distribution from policy
        pi_legal = root_node.N / (root_node.N_total - 1)  # -1 to not count exploration of the root-node itself
        temp = 0.25 + 0.99**game_id
        pi_scaled = temperature_scale(pi_legal, temp)

        # Selecet action
        action = np.random.choice(n_actions, size=1, p=pi_scaled)[0]

        # Pick move
        root_node = root_node.action_edges[action]
        S_new_obs, r, done, info = env.step(action)
        total_R += r
        # Save Data
        ER.store(S_obs, action, r, done, root_node.v, pi_legal)
        S_obs = S_new_obs
        if done:
            # Check for termination of environment
            wr_Q.put(['scalar', 'environment/steps', turns, game_id])
            wr_Q.put(['scalar', 'environment/total_reward', total_R, game_id])
            env.close()
            break

def sim_game_worker(env_maker, f_g_Q, h_Q, EX_Q, lock, game_counter, seed, MCTS_settings, MuZero_settings, experience_settings):
    np.random.seed(seed)
    n_games = MuZero_settings["N_training_games"]
    while True:
        with lock:
            game_counter.value += 1
            val = game_counter.value
        if not (val > n_games):
            sim_game(env_maker, val, seed, f_g_Q, h_Q, EX_Q, MCTS_settings, MuZero_settings, experience_settings)
        else:
            return

def sim_games(env_maker, f_model, g_model, h_model, EX_Q, MCTS_settings, MuZero_settings, experience_settings):
    number_of_processes = MCTS_settings["number_of_threads"]
    # Function for generating games
    process_workers = []
    # This is important for generating a worker with torch support
    torch.multiprocessing.set_start_method('spawn', force=True)
    f_model.eval()  # Set model for evaluating
    g_model.eval()
    h_model.eval()

    # Make queues for sending data
    gf_model_Q = Queue()  # For sending board positions to GPU
    hf_model_Q = Queue()  # For sending board positions to GPU
    conn_rec, conn_send = Pipe(False)

    # Make counter and lock
    game_counter = Value('i', 0)
    lock = Lock()

    # Make process for gpu workers
    hidden_input_size = hidden_input_size = (MCTS_settings["action_size"][0]+1,) + MCTS_settings["hidden_S_size"]
    process_workers.append(Process(target=gpu_worker, args=(gf_model_Q, hidden_input_size, MCTS_settings, g_model, f_model, True)))
    process_workers.append(Process(target=gpu_worker, args=(hf_model_Q, MCTS_settings["observation_size"], MCTS_settings, h_model, f_model, False)))
    # Start gpu and data_loader worker
    for p in process_workers:
        p.start()
    # Construct tasks for workers
    procs = []
    torch.multiprocessing.set_start_method('fork', force=True)
    for i in range(number_of_processes):
        seed = np.random.randint(int(2 ** 31))
        procs.append(Process(target=sim_game_worker,
                             args=(env_maker, gf_model_Q, hf_model_Q, EX_Q, lock, game_counter, seed, MCTS_settings,  MuZero_settings, experience_settings)))

    # Begin running games
    for p in procs:
        p.start()
    # Join processes

    # Add a loading bar
    with tqdm(total=MuZero_settings["N_training_games"]) as pbar:
        old_iter = 0
        while True:
            if any([p.is_alive() for p in procs]):  # Check if any processes is alive
                # Update loading bar
                new_iter = game_counter.value
                pbar.update(new_iter - old_iter)
                old_iter = new_iter
                # Sleep
                time.sleep(1)
            else:
                break

    for p in procs:
        p.join()

    # Close processes
    for p in process_workers:
        p.terminate()