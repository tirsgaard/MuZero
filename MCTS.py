import numpy as np
from collections import deque, Counter
from torch.multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool

class state_node:
    def __init__(self, action_size):
        self.action_edges = {}  # dictionary containing reference to child notes
        self.N_total = 1  # Total times node visited during MCTS
        self.game_over = False  # If game is over
        self.explored = False
        self.N = np.zeros(action_size)  # Total times each child node visited during MCTS
        self.N_inv = np.ones(action_size)  # Store inverse N for faster calculations
        self.Q = np.zeros(action_size)  # Average return observed for state for all actions
        self.illegal_actions = np.zeros(action_size)  # If any actions are illegal, default is none

    def explore(self, S, P, r, v):
        self.r = r  # reward from travelling from parent to this node
        self.S = S  # Dynamic state
        self.U = P  # Exploration values
        self.P = P  # Exploration values
        self.v = v  # Value of state. Needs to be stored for priority sampling in ER
        self.explored = True

    def set_illegal(self, illegal_actions):
        # Function for inserting illegal actions.
        # Input:
        #    illegal_actions: array of same size as action space, with illegal actions set to 1, else 0
        self.illegal_actions[illegal_actions.astype("bool")] = float("-inf")


def node_back_up(node_path, v, gamma, n_vl):
    # Function for backing up after a new node has been evaluated
    l = len(node_path) # Length of path
    k = l
    r = []  # List containing all rewards (r)
    for node, action, color in reversed(node_path):
        # Loop over path in reverse order for easier update of r
        node.N_total += -n_vl + 1
        node.N[action] += -n_vl + 1
        # Skip inverse since it will be updated later
        node.N_inv[action] = 1 / (node.N[action] + 1)

        taus = l-k-1 - np.arange(0, l-k)  # Reverse list of tau
        gammas = gamma**taus
        G = sum(gammas*r) + gamma**(l-k)*v
        node.W[action] += G + n_vl
        node.Q[action] = node.W[action] / node.N[action]

        r.append(node.r)
        k -= 1
    return node.Q[action]  # used for setting range of returns for PUCT


def MCTS(root_node, f_g_Q, MCTS_settings):
    n_parallel_explorations = MCTS_settings["n_parallel_explorations"]
    N_MCTS_sim = MCTS_settings["N_MCTS_sim"]
    # Define pipe for GPU process
    conn_rec, conn_send = Pipe(False)
    min_Q, max_Q = np.array(float("-Inf")), np.array(float("Inf"))

    # Begin simulations
    i = 0
    while i < N_MCTS_sim:  # Keep number of simulations below the set N
        stored_jobs = []
        empty = False
        for j in range(n_parallel_explorations):
            current_path = deque([])
            stored_jobs = select_node(root_node, min_Q, max_Q, stored_jobs, MCTS_settings)

            # Code for breaking simuntanious searches of same node
            if ((empty == True) & (stored_jobs == [])):
                break

        if ((empty == True) & (stored_jobs == [])):
            # This is the case where the first unexplored edge is an ending game, so might as well stop multi-jobs
            i += 1
            empty = False
            continue

        # Store values for updating
        S_array, r_array, P_array, v_array = expand_node(stored_jobs, f_g_Q, conn_send, conn_rec)

        # Expand tree and backtrack each node in queue
        min_Q_b, max_Q_b = backup_node(stored_jobs, S_array, r_array, P_array, v_array, MCTS_settings)
        # Update the interval of observed reward returns
        min_Q = min(min_Q_b, min_Q)
        max_Q = max(max_Q_b, max_Q)
        i += len(stored_jobs)

    return root_node

def select_node(root_node, min_Q, max_Q, stored_jobs, MCTS_settings):
    n_vl = MCTS_settings["virtual_loss"]
    c1 = MCTS_settings["c1"]
    c2 = MCTS_settings["c2"]
    current_path = deque([])
    current_node = root_node
    normalizer = max_Q-min_Q # Value for normalising
    while True:  # Continue to traverse tree until new edge is found
        temp = c1+np.log((current_node.N+c2+1)/c2)
        # Choose action
        current_node.U = current_node.P * np.sqrt(current_node.N_total)/(1+current_node.N)*temp
        Q_normed = (current_node.Q - min_Q)/normalizer
        a_chosen = np.argmax(Q_normed + current_node.U + current_node.illigal_board)
        current_node.N_total += n_vl - 1  # Add virtual loss

        # Add action and node to path and change color
        # Add current node to path
        current_path.append((current_node, a_chosen))

        ## Update stored values using virtual loss
        current_node.W[a_chosen] += -n_vl
        current_node.Q[a_chosen] = (current_node.W[a_chosen]) / (current_node.N[a_chosen] + n_vl)

        # continue based if edges are explored or not
        if current_node.N[a_chosen] != 0:  # Case for explored edge
            # Update current node, color of turn, and repeat
            current_node.N[a_chosen] += n_vl
            current_node.N_inv[a_chosen] = 1 / (current_node.N[a_chosen] + 1)

            current_node = current_node.action_edges[a_chosen]  # Select new node
            if not current_node.explored:
                # Case where same new edge is explored during parallel
                # Undo virtual loss
                for node, action, color in current_path:
                    node.N_total -= n_vl
                    node.N[action] -= n_vl
                    node.N_inv[action] = 1 / (node.N[action] + 1)
                    node.W[action] += n_vl
                    node.Q[action] = node.W[action] / node.N[action]
                empty = True
                break
            continue

        else:  # Not explored case
            # Virtual loss update could be skipped for the last node
            #   but it makes the code simpler to include it
            current_node.N[a_chosen] += n_vl
            current_node.N_inv[a_chosen] = 1 / (current_node.N[a_chosen] + 1)

            # Make new end node
            new_node = state_node()
            current_node.action_edges[a_chosen] = new_node

            # Get previous state
            S = current_node.S
            # Save values for later construction and backup of node
            stored_jobs.append([S, current_path, current_node, new_node, a_chosen])
            break
    return stored_jobs


def expand_node(stored_jobs, f_g_Q, f_g_send, f_g_rec, MCTS_settings):
    obs_size = MCTS_settings["observation_size"]
    # Store values for updating
    S_array = np.empty((len(stored_jobs), ) + (obs_size,), dtype=bool)
    a_array = np.empty((len(stored_jobs), ), dtype=int)

    k = 0
    for S, current_path, new_go_state, current_path, current_node, a_chosen in stored_jobs:
        S_array[k] = S
        a_array[k] = a_chosen
        k += 1

    # Get policy and value
    f_g_Q.put([S_array, f_g_send])
    S_array, r_array, P_array, v_array = f_g_rec.recv()
    return S_array, r_array, P_array, v_array


def backup_node(stored_jobs, S_array, r_array, P_array, v_array, MCTS_settings):
    n_vl = MCTS_settings["virtual_loss"]
    action_size = MCTS_settings["action_size"]
    gamma = MCTS_settings["gamma"]
    l = len(stored_jobs)
    Q_vals = np.zeros((l,)) # Array for storing Q-values to find max and min values for pUCT

    # Expand tree and backtrack each node in queue
    k = 0
    for S, new_go_state, current_path, leaf_node, a_chosen in stored_jobs:
        P = P_array[k]
        r = r_array[k][0]
        S = S_array[k]
        v = v_array[k][0]

        # Add explore information to leaf node
        leaf_node.explore(S, P, r, v)
        # Now back up and remove virtual loss
        Q_vals[k] = node_back_up(current_path, v, gamma, n_vl)
        k += 1  # Update k for each simulation

    min_Q = np.min(Q_vals)
    max_Q = np.max(Q_vals)
    return min_Q, max_Q

