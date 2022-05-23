from line_profiler_pycharm import profile
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
from time import time
from agents import UCB1_agent, UCB1_tuned_agent, UCB1_normal_agent, UCB2_agent, Bays2_agent, Bayes_agent
from scipy.stats import bernoulli
import math

@profile
def tree_thread(N_data_points, agents, options):

    seed = np.random.randint(0, 10**9)
    true_mu = 0.

    true_sigma = 0.1
    min_max_tree = False
    N_bandits = 5  # Number of bandits pr. leaf
    n_agents = len(agents)

    tree_shape = (N_bandits, N_bandits)
    depth = len(tree_shape)

    # Generate means and sigma for normal samples of bandits
    # bandit_mu = np.random.normal(true_mu, true_sigma, (n_sim, ) + tree_shape)
    bandit_mu = np.random.rand(*tree_shape)
    #bandit_sigma = np.full((tree_shape[0], tree_shape[1]), true_sigma)

    a_array = np.zeros((N_data_points, depth, n_agents), dtype="uint16")
    cum_regret = np.zeros((N_data_points, n_agents))
    greedy_array = np.zeros((N_data_points, n_agents))
    regret_second_array = np.zeros((N_data_points, n_agents))
    optimal_action = np.zeros((N_data_points, n_agents))
    optimal_top_action = np.zeros((N_data_points, n_agents))
    reward = np.zeros((N_data_points, n_agents))
    top_action_bias = np.zeros((N_data_points, n_agents))

    best_reward = np.max(bandit_mu)  # mean reward from best bandit

    best_bandits = bandit_mu.reshape(5, -1).max(axis=1)
    best_bandit_second = bandit_mu.max(axis=1)
    if min_max_tree:
        best_bandit_top = bandit_mu.min(axis=1).argmax()
    else:
        best_bandit_top = bandit_mu.max(axis=1).argmax()  #np.stack(np.unravel_index(np.argmax(bandit_mu.reshape(-1)), tree_shape)).transpose()
    #  Initialize array and fill with agent
    for j in range(len(agents)):
        np.random.seed(seed)
        opt = [N_bandits] + options[j]
        agent_list = []
        for d in range(depth):
            agent_shape = tree_shape[0:d]
            agent_array = np.empty((int(np.prod(agent_shape)),), dtype="object")
            # Fill array with agents
            for i in range(np.prod(agent_array.shape)):
                agent_array[i] = agents[j](*opt)  # Give arguments to agent for initialisation
            agent_array = agent_array.reshape((1, )+agent_shape)
            agent_list.append(agent_array)


        # Begin simulation
        for i in range(N_data_points):
            # Generate all actions for depths
            a_list = [[0]]
            for d in range(depth):
                if d==1 and min_max_tree:
                    action = agent_list[d][a_list][0].act_min()
                else:
                    action = agent_list[d][a_list][0].act()
                a_list.append(action)
            r = bernoulli.rvs(bandit_mu[a_list[1:]])  # np.random.normal(bandit_mu[a_list[1:]], bandit_sigma[a_list[1:]])[0]
            # Backup result
            extra_info = None
            for d in reversed(range(depth)):
                extra_info = agent_list[d][a_list[0:(d+1)]][0].backup(a_list[d+1], r, extra_info)

            # Get statistics
            a_array[i, :, j] = a_list[1:]
            # Average regret
            cum_regret[i, j] = cum_regret[i - 1, j] + best_reward - bandit_mu[a_list[1:]]
            # Greedy error
            greedy_action = np.zeros((depth, ), dtype=np.int64)
            a_list = [[0]]
            for d in range(depth):
                greedy_action[d] = int(agent_list[d][a_list][0].Q.argmax())
                a_list.append([greedy_action[d]])

            greedy_array[i, j] = bandit_mu[a_list[1:]]
            opt_action = greedy_action == best_bandits
            optimal_action[i, j] = np.all(opt_action, axis=0)
            optimal_top_action[i, j] = 1 - (agent_list[0][0].Q.argmax(axis=0) == best_bandit_top)
            regret_second_array[i, j] = best_reward - best_bandit_second[a_list[1]]
            top_action_bias[i, j] = bandit_mu[a_list[1:]] - agent_list[0][0].Q.max(axis=0)
            reward[i, j] = r

    """
    optimal_action = np.mean(optimal_action, axis=1)
    optimal_top_action = np.mean(optimal_top_action, axis=1)
    cum_regret = np.mean(cum_regret, axis=1)
    reward = np.mean(reward, axis=1)
    greedy_array = np.mean(greedy_array, axis=1)
    """
    return optimal_top_action, cum_regret, reward, greedy_array, regret_second_array, top_action_bias

def tree_simulation(N_data_points, n_sim, agents):
    true_mu = 0.
    true_sigma = 1.
    N_bandits = 5  # Number of bandits pr. leaf
    n_agents = len(agents)

    tree_shape = (N_bandits, N_bandits, N_bandits)
    depth = len(tree_shape)

    # Generate means and sigma for normal samples of bandits
    #bandit_mu = np.random.normal(true_mu, true_sigma, (n_sim, ) + tree_shape)
    bandit_mu = np.random.rand(n_sim, tree_shape[0], tree_shape[1], tree_shape[2])
    #bandit_sigma = np.full((n_sim, ) + tree_shape, true_sigma)


    a_array        = np.zeros((N_data_points, depth, n_sim, n_agents), dtype="uint16")
    cum_regret     = np.zeros((N_data_points, n_sim, n_agents))
    greedy_array   = np.zeros((N_data_points, n_sim, n_agents))
    optimal_action = np.zeros((N_data_points, n_sim, n_agents))
    optimal_top_action = np.zeros((N_data_points, n_sim, n_agents))
    reward         = np.zeros((N_data_points, n_sim, n_agents))

    best_reward = np.max(bandit_mu.reshape(n_sim, -1), axis=1)  # mean reward from best bandit
    best_bandits = np.stack(np.unravel_index(np.argmax(bandit_mu.reshape(n_sim, -1), axis=1), tree_shape)).transpose()
    for j in range(len(agents)):
        agent = agents[j](tree_shape, n_sim)
        for i in range(N_data_points):
            # Generate returns for all bandits for paral
            #s = np.random.normal(bandit_mu, bandit_sigma)
            s = bernoulli.rvs(bandit_mu)
            a = agent.act()
            r = s[range(n_sim), a[:, 0], a[:, 1], a[:, 2]]  # Take action
            agent.backup(a, r)

            a_array[i, :, :, j] = a.transpose()
            # Average regret
            cum_regret[i, :, j] = cum_regret[i - 1, :, j] + best_reward - bandit_mu[range(n_sim), a[range(n_sim), 0], a[range(n_sim), 1], a[range(n_sim), 2]]
            # Greedy error
            indexes = list(np.unravel_index(np.argmax(agent.Qs[-1].reshape(-1, n_sim), axis=0), tree_shape))
            indexes.insert(0, np.array(range(n_sim)))
            greedy_array[i, :, j] = bandit_mu[indexes]
            opt_action = a == best_bandits
            optimal_action[i, :, j] = np.all(opt_action, axis=1)
            optimal_top_action[i, :, j] = 1 - (agent.Qs[0].argmax(axis=0) == best_bandits[:, 0])
            reward[i, :, j] = r

    optimal_action = np.mean(optimal_action, axis=1)
    optimal_top_action = np.mean(optimal_top_action, axis=1)
    cum_regret = np.mean(cum_regret, axis=1)
    reward = np.mean(reward, axis=1)
    greedy_array = np.mean(greedy_array, axis=1)
    return optimal_top_action, cum_regret, reward, greedy_array



def run_simulation(N_data_points, n_sim):
    true_mu = 0.
    true_sigma = 1.
    N_bandits = 20
    mu_prior = np.array([true_mu] * N_bandits)
    sigma_prior = np.array([true_sigma] * N_bandits)
    n_agents = 3

    a_array        = np.zeros((N_data_points, n_sim, n_agents), dtype="uint16")
    cum_regret     = np.zeros((N_data_points, n_sim, n_agents))
    greedy_array   = np.zeros((N_data_points, n_sim, n_agents))
    optimal_action = np.zeros((N_data_points, n_sim, n_agents))
    reward         = np.zeros((N_data_points, n_sim, n_agents))

    # Generate bandit distributions
    bandit_mu = np.random.normal(true_mu, true_sigma, (N_bandits))#np.array([0, 1])#
    bandit_sigma = np.array([true_sigma] * N_bandits)
    # Generate returns for all bandits
    s = np.random.normal(bandit_mu, bandit_sigma, (N_data_points, n_sim, N_bandits)).transpose()

    def agent_sim(agent, agent_id):
        for i in range(N_data_points):
            a = agent.act()
            r = s[a, range(n_sim), i]  # Take action
            agent.backup(a, r)

            a_array[i, :, agent_id] = a
            # Average regret
            cum_regret[i, :, agent_id] = cum_regret[i-1, :, agent_id] + np.max(bandit_mu, axis=0) - bandit_mu[a]
            greedy_array[i, :, agent_id] = bandit_mu[np.argmax(UCB_agent.Q, axis=0)]  # If no exploration was applied
            optimal_action[i, :, agent_id] = a == np.argmax(bandit_mu)
            reward[i, :, agent_id] = r

    # Now do the same for UCB1
    UCB_agent = UCB1_agent(N_bandits, n_sim)
    agent_sim(UCB_agent, 0)

    # Now do the same for UCB2-normal
    UCB_agent = UCB2_agent(N_bandits, n_sim)
    agent_sim(UCB_agent, 1)

    # Now do the same for UCB2-normal
    UCB_agent = UCB1_tuned_agent(N_bandits, n_sim=n_sim, upper_variance=2, known_variance=True)
    agent_sim(UCB_agent, 2)

    # average over n_sims
    optimal_action = np.mean(optimal_action, axis=1)
    cum_regret = np.mean(cum_regret, axis=1)
    reward = np.mean(reward, axis=1)
    greedy_array = np.mean(greedy_array, axis=1)
    return optimal_action, cum_regret, reward, greedy_array

def run_simulation_beta(N_data_points, n_sim):
    def transform_par(alpha, beta):
        a = np.log(np.exp(alpha) - 1)
        b = np.log(np.exp(beta) - 1)
        return a, b

    def retransform_par(a, b):
        alpha = np.log(1 + np.exp(a))
        beta = np.log(1 + np.exp(b))
        return alpha, beta

    def drift_parameter(alpha, beta, var):
        a, b = transform_par(alpha, beta)
        a, b = np.random.normal(np.array([a, b]), var)
        alpha, beta = retransform_par(a, b)
        return alpha, beta


    N_bandits = 10  # 19*19  # Number of bandits
    N_agents = 4  # Number of types of agent to run

    greedy_array   = np.zeros((N_data_points, n_sim, N_agents))
    regret         = np.zeros((N_data_points, n_sim, N_agents))
    optimal_action = np.zeros((N_data_points, n_sim, N_agents))
    reward         = np.zeros((N_data_points, n_sim, N_agents))


    def agent_sim(UCB_agent, agent_id):
        # Generate bandits with uniform distance in interval [0.25-0.75]
        means = np.linspace(0.25, 0.75, N_bandits)[:, None].repeat(n_sim, 1)
        # These are the non-transformed coordinates
        alpha = means * 0 + 1
        beta = -(means - 1) / means
        var_vals = 0.1

        for i in range(N_data_points):
            a = UCB_agent.act()
            r = np.random.beta(alpha[a, range(n_sim)], beta[a, range(n_sim)])
            UCB_agent.backup(a, r)

            bandit_mean = alpha / (alpha + beta)
            regret[i, :, agent_id] = regret[i - 1, range(n_sim), agent_id] + np.max(bandit_mean, axis=0) - bandit_mean[
                a, range(n_sim)]  # Average regret
            optimal_action[i, :, agent_id] = a == np.argmax(bandit_mean, axis=0)
            reward[i, :, agent_id] = reward[i - 1, :, agent_id] + r
            greedy_array[i, :, agent_id] = bandit_mean[np.argmax(UCB_agent.Q, axis=0), range(n_sim)]
            # Update distribution
            alpha[a, range(n_sim)], beta[a, range(n_sim)] = drift_parameter(alpha[a, range(n_sim)],
                                                                            beta[a, range(n_sim)], var_vals)

    # Now do the same for UCB1
    UCB_agent = UCB1_agent(N_bandits, n_sim)
    agent_sim(UCB_agent, 0)

    # Now do the same for UCB1-normal
    UCB_agent = UCB1_normal_agent(N_bandits, n_sim)
    agent_sim(UCB_agent, 1)

    # Now do the same for UCB2
    UCB_agent = UCB2_agent(N_bandits, n_sim)
    agent_sim(UCB_agent, 2)

    # Now do the same for UCB1 self-tuning
    UCB_agent = UCB1_tuned_agent(N_bandits, n_sim)
    agent_sim(UCB_agent, 3)


    optimal_action = np.mean(optimal_action, axis=1)
    regret = np.mean(regret, axis=1)
    reward = np.mean(reward, axis=1)
    greedy_array = np.mean(greedy_array, axis=1)
    return optimal_action, regret, reward, greedy_array