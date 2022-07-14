import numpy as np
from collections import deque, Counter
from torch.multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
import graphviz
import torch.nn as nn
from models import stack_a
from helper_functions import sum_dist
import math

def Phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def phi(x):
    # Propability mass distribution of the standard normal distribution
    return 0.3989422802*math.exp(-0.5*x*x)  # Number is 1/sqrt(2*pi)


class node_base:
    def __init__(self, id):
        self.action_edges = {}  # dictionary containing reference to child notes
        self.id = id  # Not needed for algorithm, but useful for visualisation
        self.N_total = 1  # Total times node visited during MCTS
        self.game_over = False  # If game is over
        self.explored = False

    def add_vl(self, a, n_vl):
        # Currently not doing anything because of lack of knowledge of return distribution
        self.N_total += n_vl  # Add virtual loss
        self.N[a] += n_vl
        self.N_inv[a] = 1/self.N[a]

    def remove_vl(self, a, n_vl):
        # Currently not doing anything because of lack of knowledge of return distribution
        self.N_total += -n_vl
        self.N[a] += -n_vl
        # Skip inverse since it will be updated later
        self.N_inv[a] = 1 / self.N[a]

    def set_illegal(self, illegal_actions):
        # Function for inserting illegal actions.
        # Input:
        #    illegal_actions: array of same size as action space, with illegal actions set to 1, else 0
        self.illegal_actions[illegal_actions.astype("bool")] = float("-inf")


class state_node(node_base):
    def __init__(self, action_size, id):
        node_base.__init__(self, id)
        self.N = np.zeros(action_size)  # Total times each child node visited during MCTS
        self.N_inv = np.ones(action_size)  # Store inverse N for faster calculations
        self.Q = np.zeros(action_size, dtype=np.float64)  # Average return observed for state for all actions
        self.W = np.zeros(action_size, dtype=np.float64)  # Used to make update of average return Q
        self.squard_return = np.zeros(action_size, dtype=np.float64)
        self.mu = np.zeros(action_size, dtype=np.float64)
        self.sigma_squared = np.zeros(action_size, dtype=np.float64)
        self.illegal_actions = np.zeros(action_size)  # If any actions are illegal, default is none

    def explore(self, S, P, r, v):
        self.r = float(r)  # Reward from travelling from parent to this node
        self.S = S  # Dynamic state
        self.U = P  # Exploration values
        self.P = P  # Exploration values
        self.v = float(v)  # Value of state. Needs to be stored for priority sampling in ER
        self.explored = True

    def add_vl(self, a, n_vl):
        self.N_total += n_vl  # Add virtual loss
        self.N[a] += n_vl
        self.N_inv[a] = 1/self.N[a]
        self.W[a] += -n_vl
        self.Q[a] = self.W[a] / self.N[a]

    def remove_vl(self, a, n_vl):
        self.N_total += -n_vl
        self.N[a] += -n_vl
        # Skip inverse since it will be updated later
        self.N_inv[a] = 1 / self.N[a]
        self.W[a] += n_vl
        self.Q[a] = self.W[a] / self.N[a]

class bayes_state_node(node_base):
    def __init__(self, action_size, id):
        node_base.__init__(self, id)
        self.N = np.zeros(action_size)  # Total times each child node visited during MCTS
        self.N_inv = np.ones(action_size)  # Store inverse N for faster calculations
        self.Q = np.zeros(action_size)  # Average return observed for state for all actions
        self.squard_return = np.zeros(action_size, dtype=np.float64)
        self.mu = np.zeros(action_size, dtype=np.float64)
        self.sigma_squared = np.zeros(action_size, dtype=np.float64)

    def explore(self, S, Prior, r):
        self.r = float(r)  # Reward from travelling from parent to this node
        self.S = S  # Dynamic state
        self.U = Prior  # Exploration values
        self.mu = Prior[:, 0]
        self.sigma_squared = Prior[:, 1]
        self.explored = True
        mu, sigma = self.calc_max_dist()
        return mu, sigma

    def backup(self, action, value_dist, n_vl=0):
        self.N_total += 1
        self.N[action] += 1
        # Skip inverse since it will be updated later
        self.N_inv[action] = 1 / self.N[action]

        mu, sigma_squared = value_dist
        self.Q[action] = mu
        self.mu[action] = mu
        self.sigma_squared[action] = sigma_squared

        # Calculate own bandit distribution
        mu, sigma = self.calc_max_dist()
        return mu, sigma

    def calc_max_dist(self):
        mu = self.Q[0]
        sigma = self.sigma_squared[0]
        for i in range(1, self.Q.shape[0]):
            mu, sigma = self.max_dist(mu, self.Q[i], sigma, self.sigma_squared[i])
        return mu, sigma

    def max_dist(self, mu1, mu2, sigma1, sigma2):
        # Note sigma needs to be squared
        rho = 0
        sigma_m = np.sqrt(sigma1+sigma2-2*rho*np.sqrt(sigma1*sigma2))
        alpha = (mu1-mu2)/sigma_m

        Phi_alpha = Phi(alpha)  # Store to avoid
        phi_alpha = phi(alpha)
        F1 = alpha*Phi_alpha + phi_alpha
        F2 = alpha*alpha*Phi_alpha*(1-Phi_alpha) + (1-2*Phi_alpha)*alpha*phi_alpha - phi_alpha*phi_alpha
        mu = mu2 + sigma_m*F1
        sigma = sigma2 + (sigma1-sigma2)*Phi_alpha + sigma_m * sigma_m * F2
        return mu, sigma


class bayes_dist_node(node_base):
    def __init__(self, action_size, id, support):
        node_base.__init__(self, id)
        self.N = np.zeros(action_size)  # Total times each child node visited during MCTS
        self.N_inv = np.ones(action_size)  # Store inverse N for faster calculations
        self.n_support = support.shape[0]
        self.support_squared = self.support*self.support  # This needs to be calculated for variance
        self.Q = np.zeros(action_size, self.n_support)  # Average return observed for state for all actions
        self.mu = np.zeros(action_size, dtype=np.float64)
        self.sigma_squared = np.zeros(action_size, dtype=np.float64)

    def explore(self, S, Prior, r):
        self.r = float(r)  # Reward from travelling from parent to this node
        self.S = S  # Dynamic state
        self.U = Prior  # Exploration values
        self.mu = Prior
        self.explored = True
        Q = self.calc_max_dist()
        return Q

    def get_mean_variance(self, a):
        mu = np.sum(self.Q[a]*self.support)
        sigma_squared = np.sum(self.Q[a]*self.support_squared) - mu*mu
        return mu, sigma_squared

    def backup(self, action, value_dist, n_vl=0):
        self.N_total += 1
        self.N[action] += 1
        # Skip inverse since it will be updated later
        self.N_inv[action] = 1 / self.N[action]

        mu, sigma_squared = value_dist
        self.Q[action] = mu
        self.mu[action], self.sigma_squared[action] = self.get_mean_variance(self, a)

        # Calculate own bandit distribution
        Q = self.calc_max_dist()
        return Q

    def calc_max_dist(self):
        Q = self.Q[0]
        for i in range(1, self.Q.shape[0]):
            Q = self.max_dist(Q, self.Q[i])
        return Q

    def max_dist(self, Q1, Q2):
        # Use cumsum for O(N) scaling. Alternative is outer product
        cdf1 = Q1.cumsum(axis=0)
        cdf2 = Q2.cumsum(axis=0)
        Q = cdf1 * Q2 + cdf2 * Q1 - Q1 * Q2
        return Q


class min_max_vals:
    def __init__(self, min_val=None, max_val=None):
        # Default values have been reversed to skip normalisation until 2 values have been found
        self.max = -float("inf") if max_val is None else max_val
        self.min = float("inf") if min_val is None else max_val

    def update(self, Q):
        self.max = Q if (Q > self.max) else self.max
        self.min = Q if (Q < self.min) else self.min

    def norm(self, Q):
        norm_Q = Q
        if self.min < self.max:  # Skip until normalisation until two values have been observed
            norm_Q = (Q - self.min) / (self.max - self.min)
        return norm_Q


def generate_root(S_obs, h_Q, h_send, h_rec, MCTS_settings):
    h_Q.put([S_obs[None], h_send])  # Get hidden state of observation
    S, P, v = h_rec.recv()
    if MCTS_settings["bayesian"]:
        root_node = bayes_state_node(MCTS_settings["action_size"], 0)
        root_node.explore(S[0], P[0], 0.0)
    else:
        root_node = state_node(MCTS_settings["action_size"], 0)
        root_node.explore(S[0], P[0], 0.0, v[0])
    return root_node


def MCTS(root_node, f_g_Q, MCTS_settings):
    n_parallel_explorations = MCTS_settings["n_parallel_explorations"]
    N_MCTS_sim = MCTS_settings["N_MCTS_sim"]
    # Define pipe for GPU process
    conn_rec, conn_send = Pipe(False)
    min_val = MCTS_settings["min_val"]
    max_val = MCTS_settings["max_val"]
    normalizer = min_max_vals(min_val, max_val)

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

        if MCTS_settings["bayesian"]:
            weights = current_node.Q + np.sqrt(8*np.log(current_node.N_total)*current_node.sigma_squared)
            a_chosen = np.argmax(weights + np.random.rand(weights.shape[0]) * 1e-10)  # random noise faster than permutation to break tiebreaker
            #a_chosen = np.argmax(np.random.normal(current_node.Q, np.sqrt(current_node.sigma_squared)))
        else:
            temp = c1+np.log((current_node.N_total+c2)/c2)  # +1 is removed as as root node counts itself
            # Choose action
            current_node.U = current_node.P * np.sqrt(current_node.N_total - 1)*temp/(1+current_node.N)  # -1 because of root node
            Q_normed = normalizer.norm(current_node.Q)
            weights = Q_normed + current_node.U + current_node.illegal_actions
            a_chosen = np.argmax(weights + np.random.rand(weights.shape[0])*1e-10)  # random noise faster than permutation to break tiebreaker

        # Add current node to path
        current_path.append((current_node, a_chosen))

        ## Update stored values using virtual loss
        current_node.add_vl(a_chosen, n_vl)

        # continue based if edges are explored or not
        if current_node.N[a_chosen] != n_vl:  # Case for explored edge
            # Update current node, color of turn, and repeat
            current_node = current_node.action_edges[a_chosen]  # Select new node
            if not current_node.explored:
                # Case where same new edge is explored during parallel
                # Undo virtual loss
                for node, action in current_path:
                    node.undo_vl(action, n_vl)
                break
            continue

        else:  # Not explored case
            # Virtual loss update could be skipped for the last node
            #   but it makes the code simpler to include it
            # Make new end node
            if MCTS_settings["bayesian"]:
                new_node = bayes_state_node(MCTS_settings["action_size"], leaf_number)
            else:
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

    Sa_array = np.concatenate(Sa_array, axis=0)
    # Get policy and value
    f_g_Q.put([Sa_array, f_g_send])
    S_array, r_array, P_array, v_array = f_g_rec.recv()
    return S_array, r_array, P_array, v_array


def backup_node(stored_jobs, S_array, r_array, P_array, v_array, normalizer, MCTS_settings):
    n_vl = MCTS_settings["virtual_loss"]
    gamma = MCTS_settings["gamma"]

    # Expand tree and backtrack each node in queue
    k = 0
    for S, current_path, current_node, leaf_node, a_chosen in stored_jobs:
        P = P_array[k]
        r = r_array[k]
        S = S_array[k]
        v = v_array[k]

        # Now back up and remove virtual loss
        if MCTS_settings["bayesian"]:
            # Add explore information to leaf node
            v_max_dist = leaf_node.explore(S, P, r)
            bayes_back_up(current_path, v_max_dist, gamma, n_vl, normalizer)
        else:
            v_max_dist = leaf_node.explore(S, P, float(r), float(v))
            node_back_up(current_path, v, gamma, n_vl, normalizer)
        k += 1  # Update k for each simulation


def node_back_up(node_path, v, gamma, n_vl, normalizer):
    # Function for backing up after a new node has been evaluated
    l = len(node_path)  # Length of path
    k = l
    r = []  # List containing all rewards (r)
    r_cum = 0
    v_cum = v
    for node, action in reversed(node_path):
        # Loop over path in reverse order for easier update of r
        node.undo_vl(action, n_vl)
        node.N_total += 1
        node.N[action] += 1
        # Skip inverse since it will be updated later
        node.N_inv[action] = 1 / node.N[action]
        r_cum = gamma*r_cum + node.action_edges[action].r
        v_cum = gamma*v_cum
        node.W[action] += r_cum + v_cum
        node.Q[action] = node.W[action] / node.N[action]
        normalizer.update(node.Q[action])  # Update lowest and highest Q-values observed in tree
        r.append(node.r)
        k -= 1
    return


def bayes_back_up(node_path, v_dist, gamma, n_vl, normalizer):
    # Function for backing up after a new node has been evaluated
    mu, sigma_squared = v_dist
    for node, action in reversed(node_path):
        # Loop over path in reverse order for easier update of r
        node.remove_vl(action, n_vl)
        # Scale v dist with gamma
        mu = gamma * node.action_edges[action].r + gamma * gamma * mu
        sigma_squared = sigma_squared * gamma ** 4  # The variance also needs to be scaled accordingly
        # Backup and get new values
        mu, sigma_squared = node.backup(action, [mu, sigma_squared])
    return


def bayes_dist_back_up(node_path, v_dist, n_vl, contractor, add_dist_mat):
    # Function for backing up after a new node has been evaluated
    for node, action in reversed(node_path):
        # Loop over path in reverse order for easier update of r
        node.remove_vl(action, n_vl)
        # Scale v dist with gamma
        v_dist = sum_dist(contractor.contract(node.action_edges[action].r), contractor.contract_twice(v_dist), add_dist_mat)
        # Backup and get new values
        v_dist = node.backup(action, v_dist)
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
                    + "\n Q: " + str(np.round(Q_val, decimals=4)) \
                    + "\n Q_norm: " + str(np.round(Q_norm, decimals=4)) \
                    + "\n W: " + str(np.round(parent_node.W[action], decimals=4)) \
                    + "\n v: " + str(np.round(child.v, decimals=4))
        tree.node(str(id[0]), node_text, color=str(int(Q_norm*10+1)), colorscheme="rdylgn11")
        tree.edge(str(parent_id), str(id[0]),
                  penwidth=str(thick_get(visits)),
                  label="a: " + str(action) + "\n N(a): " + str(int(visits)) + "\n r: " + str(np.round(child.r, decimals=4)),
                  color=str(int(Q_norm * 10 + 1)),
                  colorscheme= "rdylgn11",
                  arrowhead='none')
        iterate_tree(tree, child, id, thick_get, normalizer)
    return tree

def gradient_clipper(model: nn.Module) -> nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad * 0.5)
    return model

class gamma_contract:
    def __init__(self, gamma, support):
        self.zero_index = np.argmax(support == 0.)
        assert(support[self.zero_index] == 0.)
        n_support = support.shape[0]
        self.identity = np.ones((n_support,))
        self.transfer = np.zeros((n_support,))
        # Also do this for gamma**2
        self.identity_twice = np.ones((n_support,))
        self.transfer_twice = np.zeros((n_support,))

        for i in range(self.zero_index):
            assert (support[i + 1] > support[i] * gamma)  # Check transfered value is in-between values
            self.transfer[i] = self.calc_coef(support[i + 1], support[i], gamma)
            assert (support[i + 1] > support[i] * gamma*gamma)  # Check transfered value is in-between values
            self.transfer_twice[i] = self.calc_coef(support[i + 1], support[i], gamma*gamma)

        for i in reversed(range(self.zero_index + 1, n_support)):
            assert (support[i - 1] < support[i] * gamma)  # Check transfered value is in-between values
            self.transfer[i] = self.calc_coef(support[i - 1], support[i], gamma)

            assert (support[i - 1] < support[i] * gamma*gamma)  # Check transfered value is in-between values
            self.transfer_twice[i] = self.calc_coef(support[i - 1], support[i], gamma*gamma)

        self.identity -= self.transfer
        self.identity_twice -= self.transfer_twice

    def contract(self, dist):
        dist2 = dist * self.identity
        end = self.zero_index + 1
        dist2[1:end] += dist[0:self.zero_index] * self.transfer[0:self.zero_index]
        dist2[self.zero_index:-1] += dist[end:] * self.transfer[end:]
        return dist2

    def contract_twice(self, dist):
        dist2 = dist * self.identity_twice
        end = self.zero_index + 1
        dist2[1:end] += dist[0:self.zero_index] * self.transfer_twice[0:self.zero_index]
        dist2[self.zero_index:-1] += dist[end:] * self.transfer_twice[end:]
        return dist2

    def calc_coef(self, low_val, high_val, gamma):
        trans = high_val * (gamma - 1) / (low_val - high_val)
        return trans