import numpy as np
from collections import deque, Counter
from torch.multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
import graphviz
from models import stack_a
class state_node:
    def __init__(self, action_size, id):
        self.action_edges = {}  # dictionary containing reference to child notes
        self.id = id  # Not needed for algorithm, but usefull for visualisation
        self.N_total = 1  # Total times node visited during MCTS
        self.game_over = False  # If game is over
        self.explored = False
        self.N = np.zeros(action_size)  # Total times each child node visited during MCTS
        self.N_inv = np.ones(action_size)  # Store inverse N for faster calculations
        self.Q = np.zeros(action_size)  # Average return observed for state for all actions
        self.W = np.zeros(action_size, dtype=np.float64)  # Used to stabilise update of average return Q
        self.illegal_actions = np.zeros(action_size)  # If any actions are illegal, default is none

    def explore(self, S, P, r, v):
        self.r = r  # Reward from travelling from parent to this node
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

class min_max_vals:
    def __init__(self):
        self.max = -float("inf")  # Values have been reversed to skip normalisation until 2 values have been found
        self.min = float("inf")

    def update(self, Q):
        self.max = Q if Q > self.max else self.max
        self.min = Q if Q < self.min else self.min

    def norm(self, Q):
        norm_Q  = Q
        if self.min < self.max:  # Skip until normalisation until two values have been observed
            norm_Q = (Q - self.min) / (self.max - self.min)
        return norm_Q


def generate_root(S_obs, h_Q, f_g_Q, h_send, h_rec, f_g_send, f_g_rec, MCTS_settings):
    root_node = state_node(MCTS_settings["action_size"], 0)
    h_Q.put([S_obs[None], h_send])  # Get hidden state of observation
    S, P, v = h_rec.recv()
    #stored_jobs = [[S, [], [], root_node, a]]
    #S_array, u_array, P_array, v_array = expand_node(stored_jobs, f_g_Q, f_g_send, f_g_rec, MCTS_settings)
    #normalizer = min_max_vals() # This will be thrown away, as the values found here are not used
    #backup_node(stored_jobs, S, u_array, P, v, normalizer, MCTS_settings)
    root_node.explore(S, P, 0, v)
    return root_node


def node_back_up(node_path, v, gamma, n_vl, normalizer):
    # Function for backing up after a new node has been evaluated
    l = len(node_path)  # Length of path
    k = l
    r = []  # List containing all rewards (r)
    r_cum = 0
    v_cum = v
    for node, action in reversed(node_path):
        # Loop over path in reverse order for easier update of r
        node.N_total += -n_vl + 1
        node.N[action] += -n_vl + 1
        # Skip inverse since it will be updated later
        node.N_inv[action] = 1 / node.N[action]

        """
        taus = l-k-1 - np.arange(0, l-k)  # Reverse list of tau
        gammas = gamma**taus
        G = sum(gammas*r) + gamma**(l-k)*v
        node.W[action] += G + n_vl
        """
        r_cum = gamma*r_cum + node.action_edges[action].r
        v_cum = gamma*v_cum
        node.W[action] += r_cum + v_cum + n_vl
        node.Q[action] = node.W[action] / node.N[action]
        normalizer.update(node.Q[action])  # Update lowest and highest Q-values observed in tree
        r.append(node.r)
        k -= 1
    return


def MCTS(root_node, f_g_Q, MCTS_settings):
    n_parallel_explorations = MCTS_settings["n_parallel_explorations"]
    N_MCTS_sim = MCTS_settings["N_MCTS_sim"]
    # Define pipe for GPU process
    conn_rec, conn_send = Pipe(False)
    normalizer = min_max_vals()

    # Begin simulations
    i = 0
    while i < N_MCTS_sim:  # Keep number of simulations below the set N
        stored_jobs = []
        for j in range(n_parallel_explorations):
            job = select_node(root_node, normalizer, i + j, MCTS_settings)
            # Code for breaking simuntanious searches of same node
            if (job == []):
                break
            stored_jobs.append(job)

        # Store values for updating
        S_array, r_array, P_array, v_array = expand_node(stored_jobs, f_g_Q, conn_send, conn_rec, MCTS_settings)
        # Expand tree and backtrack each node in queue
        backup_node(stored_jobs, S_array, r_array, P_array, v_array, normalizer, MCTS_settings)
        # Update the interval of observed reward returns
        i += len(stored_jobs)

    return root_node, normalizer


def select_node(root_node, normalizer, leaf_number, MCTS_settings):
    n_vl = MCTS_settings["virtual_loss"]
    c1 = MCTS_settings["c1"]
    c2 = MCTS_settings["c2"]
    job = []
    current_path = deque([])
    current_node = root_node
    while True:  # Continue to traverse tree until new edge is found
        temp = c1+np.log((current_node.N_total+c2+1)/c2)
        # Choose action
        current_node.U = current_node.P * np.sqrt(current_node.N_total)*temp/(1+current_node.N)
        Q_normed = normalizer.norm(current_node.Q)
        a_chosen = np.argmax(Q_normed + current_node.U + current_node.illegal_actions)

        # Add current node to path
        current_path.append((current_node, a_chosen))

        ## Update stored values using virtual loss
        current_node.N_total += n_vl  # Add virtual loss
        current_node.N[a_chosen] += n_vl
        current_node.N_inv[a_chosen] = 1/current_node.N[a_chosen]
        current_node.W[a_chosen] += -n_vl
        current_node.Q[a_chosen] = current_node.W[a_chosen] / current_node.N[a_chosen]


        # continue based if edges are explored or not
        if current_node.N[a_chosen] != n_vl:  # Case for explored edge
            # Update current node, color of turn, and repeat
            current_node = current_node.action_edges[a_chosen]  # Select new node
            if not current_node.explored:
                # Case where same new edge is explored during parallel
                # Undo virtual loss
                for node, action in current_path:
                    node.N_total -= n_vl
                    node.N[action] -= n_vl
                    node.N_inv[action] = 1 / node.N[action]
                    node.W[action] += n_vl
                    node.Q[action] = node.W[action] / node.N[action]
                break
            continue

        else:  # Not explored case
            # Virtual loss update could be skipped for the last node
            #   but it makes the code simpler to include it
            # Make new end node
            new_node = state_node(MCTS_settings["action_size"], leaf_number)
            current_node.action_edges[a_chosen] = new_node

            # Get previous state
            S = current_node.S
            # Save values for later construction and backup of node
            job = [S, current_path, current_node, new_node, a_chosen]
            break
    return job


def expand_node(stored_jobs, f_g_Q, f_g_send, f_g_rec, MCTS_settings):
    hidden_S_size = MCTS_settings["hidden_S_size"]
    action_size = MCTS_settings["action_size"]
    # Store values for updating
    Sa_array = []

    for S, current_path, current_node, new_node, a_chosen in stored_jobs:
        Sa_array.append(stack_a(S, a_chosen, hidden_S_size, action_size))

    Sa_array = np.stack(Sa_array, axis=0)
    # Get policy and value
    f_g_Q.put([Sa_array, f_g_send])
    S_array, r_array, P_array, v_array = f_g_rec.recv()
    return S_array, r_array, P_array, v_array


def backup_node(stored_jobs, S_array, r_array, P_array, v_array, normalizer, MCTS_settings):
    n_vl = MCTS_settings["virtual_loss"]
    action_size = MCTS_settings["action_size"]
    gamma = MCTS_settings["gamma"]
    l = len(stored_jobs)

    # Expand tree and backtrack each node in queue
    k = 0
    for S, current_path, current_node, leaf_node, a_chosen in stored_jobs:
        P = P_array[k]
        r = r_array[k][0]
        S = S_array[k]
        v = v_array[k][0]

        # Add explore information to leaf node
        leaf_node.explore(S, P, r, v)
        # Now back up and remove virtual loss
        node_back_up(current_path, v, gamma, n_vl, normalizer)
        k += 1  # Update k for each simulation

    return


def verify_nodes(node, MCTS_settings):
    try:
        assert(np.sum(node.N) == (node.N_total-1))  # -1 to account for initial exploration
    except AssertionError:
        print("N was: ")
        print(np.sum(node.N))
        print("N_total was: ")
        print(node.N_total-1)
        raise AssertionError

    for action in node.action_edges:
        w = node.W[action]
        child = node.action_edges[action]
        verify_w(w, child, MCTS_settings)
        verify_nodes(child, MCTS_settings)

def verify_w(w, node, MCTS_settings):
    gamma = MCTS_settings["gamma"]
    w_sum = node.N_total*node.r+gamma*node.v
    for action in node.action_edges:
        # Sum over all values W of children
        w_sum += gamma*node.W[action]
    try:
        assert(np.isclose(w, w_sum))
    except AssertionError:
        print("w was: ")
        print(w)
        print("w_sum was: ")
        print(w_sum)
        raise AssertionError


def map_tree(root_node, normalizer, game_id):
    tree = graphviz.Digraph(comment='MCT')
    id = [0]
    tree.node(str(id[0]), str(id[0]))
    max_thick = 10
    min_thick = 0.1
    increase_thick = (max_thick-min_thick)/(root_node.N_total-1)
    thick_get = lambda n: min_thick + increase_thick*n
    tree = iterate_tree(tree, root_node, id, thick_get, normalizer)
    tree.render('MCT_graphs/' + str(game_id) + '_MCTS' + '.gv').replace('\\', '/')
    'MCT_graphs/' + str(game_id) + '_MCTS' + '.gv.pdf'
    return tree

def iterate_tree(tree, parent_node, id, thick_get, normalizer):
    parent_id = id[0]
    job_list = list(parent_node.action_edges)
    for action in job_list:
        id[0] += 1
        child = parent_node.action_edges[action]
        visits = parent_node.N[action]
        Q_val = parent_node.Q[action]
        Q_norm = normalizer.norm(Q_val)
        node_text = str(child.id+1) \
                    + "\n Q: " + str(np.round(Q_val, decimals=8)) \
                    + "\n Q_norm: " + str(np.round(Q_norm, decimals=8)) \
                    + "\n W: " + str(np.round(parent_node.W[action], decimals=8)) \
                    + "\n v: " + str(np.round(child.v, decimals=8))
        tree.node(str(id[0]), node_text, color=str(int(Q_norm*10+1)), colorscheme="rdylgn11")
        tree.edge(str(parent_id), str(id[0]),
                  penwidth=str(thick_get(visits)),
                  label="a: " + str(action) + "\n N(a): " + str(int(visits)) + "\n r: " + str(np.round(child.r, decimals=8)),
                  color=str(int(Q_norm * 10 + 1)),
                  colorscheme= "rdylgn11",
                  arrowhead='none')
        iterate_tree(tree, child, id, thick_get, normalizer)
    return tree