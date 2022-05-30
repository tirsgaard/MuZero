import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import time
from agents import UCB1_agent, UCB1_tuned_agent, UCB1_normal_agent, UCB2_agent, Bays2_agent, Bayes_agent, UCB1_agent_tree, Bays_agent_Gauss_UCT2, Bays_agent_vector_UCT2, Bays_agent_Gauss_beta
from simulations import run_simulation, run_simulation_beta, tree_simulation, tree_thread, explore_tree


if __name__ == '__main__':
    np.random.seed(2)
    n_threads = 4
    N_rep = n_threads*500  # Number of repeat simulations over parallel workers for reducing variance
    n_sim = 5  # Number of in-thread simulations
    N_data_points = 100  # Number of steps to go
    agent_opt1 = [2, "PUCB", 2, 0]
    agent_opt2 = [2, "PUCB", 2, 0.3]
    agent_opt3 = [2, "PUCB", 2, 0.3]
    agent_opt4 = [2, "PUCB", 2, 0.5]
    agent_opt5 = [2, "MuZero", 2, 0.5]
    agent_opt6 = [2, "UCB1", 2, 0.5]

    option_list = [agent_opt3, agent_opt4, agent_opt5, agent_opt6]
    labels = ["PUCB, Noise 0.1: Temp: 2", "PUCB Noise 0.5: Temp: 2", "MuZero Noise 0.5: Temp: 2", "UCB1"]
    agents = [UCB1_agent, UCB1_agent, UCB1_agent, UCB1_agent]

    t = time()

    test = explore_tree(N_data_points, agents, option_list)

    p = Pool(n_threads)
    results = p.starmap(explore_tree, iterable=[(N_data_points, agents, option_list)] * N_rep)
    p.close()
    p.join()

    N_units = float(N_rep * n_sim * N_data_points)
    time_spent = time() - t
    print("Number of units was: " + str(N_units))
    print("It took: " + str(time_spent))
    print("Time pr. unit " + str(time_spent / N_units))

    # Join results
    optimal_actions = []
    regrets = []
    rewards = []
    greedy_arrays = []
    regret_second_arrays = []
    top_bias_arrays = []

    for optimal_action, regret, reward, greedy_array, regret_second, top_bias in results:
        optimal_actions.append(optimal_action)
        regrets.append(regret)
        rewards.append(reward)
        greedy_arrays.append(greedy_array)
        regret_second_arrays.append(regret_second)
        top_bias_arrays.append(top_bias)

    optimal_actions = np.array(optimal_actions)
    regrets = np.array(regrets)
    rewards = np.array(rewards)
    greedy_arrays = np.array(greedy_arrays)
    regret_second = np.array(regret_second_arrays)
    top_bias = np.array(top_bias_arrays)


    # Time for plots
    fig, ax = plt.subplots(2, 3)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]
    ax4 = ax[1, 1]
    ax5 = ax[1, 2]

    #['UCB1', "UCB2", "UCB1_tuned"]
    ax1.plot(np.mean(optimal_actions, axis=0), label=labels)
    ax1.legend()

    ax1.set_title("Error on action selected")
    ax1.set_yscale("log")
    ax1.set_ylim(0.1, 1)
    #plt.xscale("log")
    #plt.savefig("Opt_act.png")
    #plt.show()

    ax2.plot(np.mean(regrets, axis=0), label=labels)
    ax2.legend()
    ax2.set_title("Cumulative regret")
    #plt.xscale("log")
    #plt.savefig("Cum_regret.png")
    #plt.show()

    ax3.plot(np.mean(regret_second, axis=0), label=labels)
    ax3.legend()
    ax3.set_title("Regret top-level decision")
    #ax3.set_ylim(0.007, 0.2)
    ax3.set_yscale("log")
    #ax3.set_xscale("log")
    #plt.savefig("Cum_rew.png")
    #plt.show()

    ax4.plot(np.mean(greedy_arrays, axis=0), label=labels)
    ax4.legend()
    ax4.set_title("Greedy reward")

    ax5.plot(np.mean(top_bias, axis=0), label=labels)
    ax5.legend()
    ax5.set_title("Greedy top reward bias")

    # plt.xscale("log")
    #plt.savefig("Greedy_rew.png")
    plt.savefig("results.png")
    plt.show()