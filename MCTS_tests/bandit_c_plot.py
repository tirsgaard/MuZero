import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from time import time
from agents import UCB1_agent, UCB1_agent_tree, Bays_agent_vector_UCT2, Bays_agent_Gauss_beta
from simulations import run_simulation, run_simulation_beta, tree_simulation, tree_thread, explore_tree, explore_parallel_noise_tree


if __name__ == '__main__':
    np.random.seed(2)
    n_threads = cpu_count()//2  # To account for virtual cores
    N_rep = n_threads*40  # Number of repeat simulations over parallel workers for reducing variance
    n_sim = 5  # Number of in-thread simulations
    N_data_points = 50  # Number of steps to go

    n_c_points = 5
    c_range = np.linspace(0, 1, n_c_points)  #10**(np.linspace(-3, 1, n_c_points))
    agent_opt1 = [2, "UCB1", 1, 0]
    agent_opt2 = [2, True, "no_context", 1, 0]
    agent_opt3 = [2, "PUCB", 1, 0]
    agent_opt4 = [2, "alphaZero", 1, 0]
    agent_opt5 = [2, "MuZero", 1, 0]
    agent_opt6 = [2, True, "muzero_context", 1, 0]
    agent_opt7 = [2, True, "PUCT_context", 1, 0]
    agent_opt8 = [2, True, "UCT1_context", 1, 0]
    agent_opt9 = [2, True, "thompson", 1, 0]

    option_list = [agent_opt1, agent_opt2, agent_opt3, agent_opt4, agent_opt5, agent_opt6, agent_opt7, agent_opt8, agent_opt9]
    labels = ["UCB", "Uninformed Bayes UCB", "PUCB", "AlphaZero", "MuZero", "MuZero context", "PUCT_context", "informed Bayes UCB", "informed Thompson"]
    agents = [UCB1_agent, Bays_agent_Gauss_beta, UCB1_agent, UCB1_agent, UCB1_agent, Bays_agent_Gauss_beta, Bays_agent_Gauss_beta, Bays_agent_Gauss_beta, Bays_agent_Gauss_beta]

    assert (len(option_list) == len(labels))
    assert (len(option_list) == len(agents))

    n_agents = len(option_list)
    c_optimal_actions = np.zeros((n_c_points, N_rep, N_data_points, n_agents))
    c_regrets = np.zeros((n_c_points, N_rep, N_data_points, n_agents))
    c_rewards = np.zeros((n_c_points, N_rep, N_data_points, n_agents))
    c_greedy_arrays = np.zeros((n_c_points, N_rep, N_data_points, n_agents))
    c_regret_second_arrays = np.zeros((n_c_points, N_rep, N_data_points, n_agents))
    c_top_bias_arrays = np.zeros((n_c_points, N_rep, N_data_points, n_agents))

    t = time()
    for i in range(n_c_points):
        p = Pool(n_threads)
        test = explore_parallel_noise_tree(N_data_points, agents, 1, 10, c_range[i], option_list)
        results = p.starmap(explore_parallel_noise_tree, iterable=[(N_data_points, agents, 1, 10, c_range[i], option_list)] * N_rep)
        p.close()
        p.join()

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

        c_optimal_actions[i, :] = np.array(optimal_actions)
        c_regrets[i, :] = np.array(regrets)
        c_rewards[i, :] = np.array(rewards)
        c_greedy_arrays[i, :] = np.array(greedy_arrays)
        c_regret_second_arrays[i, :] = np.array(regret_second_arrays)
        c_top_bias_arrays[i, :] = np.array(top_bias_arrays)



    N_units = float(N_rep * n_sim * N_data_points)
    time_spent = time() - t
    print("Number of units was: " + str(N_units))
    print("It took: " + str(time_spent))
    print("Time pr. unit " + str(time_spent / N_units))

    # Time for plots

    def plot(ax, c_array, label, title, range_list, log_y=False):
        j = 0
        for ranges in range_list:
            vals = c_array[:, :, ranges].mean(axis=1).mean(axis=1)
            sd = np.sqrt(c_array[:, :, ranges].mean(axis=2).var(axis=1)/ N_rep)
            label_name = label + " step " + str(ranges[0]) + "-" + str(ranges[-1])
            ax.plot(c_range, vals, label=label_name, color='C' + str(j))
            ax.plot(c_range, vals + 2 * sd, '--', color='C' + str(j))
            ax.plot(c_range, vals - 2 * sd, '--', color='C' + str(j))
            j += 1

        ax.set_title(title)
        ax.set_xlabel("c")
        ax.set_ylim(0.1, 1)
        x1, x2, y1, y2 = plt.axis()
        ax.axis((x1, x2, 0.1, 1))
        #if log_y:
        #    ax.set_yscale("log")
        ax.set_xscale("log")


    def compare_plot(ax, c_array, title, ranges, log_y=False):
        vals = c_array[:, :, ranges].mean(axis=1).mean(axis=1)
        sd = np.sqrt(c_array[:, :, ranges].mean(axis=2).var(axis=1) / N_rep)
        for i in range(n_agents):
            label_name = labels[i]
            ax.plot(c_range, vals[:, i], label=label_name, color='C' + str(i))
            ax.plot(c_range, vals[:, i] + 2 * sd[:, i], '--', color='C' + str(i))
            ax.plot(c_range, vals[:, i] - 2 * sd[:, i], '--', color='C' + str(i))
        ax.legend(fontsize='xx-small')
        ax.set_title(title)
        ax.set_xlabel("c")
        if log_y:
            ax.set_yscale("log")
        ax.set_xscale("log")


    levels = [range(45, 50), range(15, 25)]
    # Compare different levels for each agent
    for i in range(n_agents):
        fig, ax = plt.subplots(2, 3)
        fig.suptitle(labels[i])
        ax1 = ax[0, 0]
        ax2 = ax[0, 1]
        ax3 = ax[1, 0]
        ax4 = ax[1, 1]
        ax5 = ax[1, 2]
        plot(ax1, c_optimal_actions[:, :, :, i], labels[i], "Error on action selected", levels, log_y=True)
        plot(ax2, c_regrets[:, :, :, i], labels[i], "Cumulative regret", levels, log_y=True)
        plot(ax3, c_regret_second_arrays[:, :, :, i], labels[i], "Regret top-level decision", levels, log_y=True)
        plot(ax4, c_greedy_arrays[:, :, :, i], labels[i], "Greedy reward", levels, log_y=True)
        plot(ax5, c_top_bias_arrays[:, :, :, i], labels[i], "Greedy top reward bias", levels, log_y=False)
        fig.set_figheight(8)
        fig.set_figwidth(15)
        plt.savefig("results_" + labels[i] + ".pdf")
        #plt.show()

    # Compare each agent at different level
    for ranges in levels:
        fig, ax = plt.subplots(2, 3)
        fig.suptitle("Comparison at step " + str(ranges[0]) + "-" + str(ranges[-1]))
        ax1 = ax[0, 0]
        ax2 = ax[0, 1]
        ax3 = ax[1, 0]
        ax4 = ax[1, 1]
        ax5 = ax[1, 2]
        compare_plot(ax1, c_optimal_actions, "Error on action selected", ranges, log_y=True)
        compare_plot(ax2, c_regrets, "Cumulative regret", ranges, log_y=False)
        compare_plot(ax3, c_regret_second_arrays, "Regret top-level decision", ranges, log_y=True)
        compare_plot(ax4, c_greedy_arrays, "Greedy reward", ranges, log_y=False)
        compare_plot(ax5, c_top_bias_arrays, "Greedy top reward bias", ranges, log_y=False)

        fig.set_figheight(8)
        fig.set_figwidth(15)
        plt.savefig("results_compare_" + str(ranges[0]) +"-" + str(ranges[-1]) + ".pdf")
        #plt.show()

    # Compare with zoom with each agent at different level
    for ranges in levels:
        fig, ax = plt.subplots(2, 3)
        fig.suptitle("Comparison at step " + str(ranges[0]) + "-" + str(ranges[-1]))
        ax1 = ax[0, 0]
        ax2 = ax[0, 1]
        ax3 = ax[1, 0]
        ax4 = ax[1, 1]
        ax5 = ax[1, 2]
        compare_plot(ax1, c_optimal_actions, "Error on action selected", ranges, log_y=True)
        compare_plot(ax2, c_regrets, "Cumulative regret", ranges, log_y=False)
        compare_plot(ax3, c_regret_second_arrays, "Regret top-level decision", ranges, log_y=True)
        compare_plot(ax4, c_greedy_arrays, "Greedy reward", ranges, log_y=False)
        compare_plot(ax5, c_top_bias_arrays, "Greedy top reward bias", ranges, log_y=False)
        ax1.set_ylim([0.2, 1])
        ax3.set_ylim([0.01, 0.2])

        fig.set_figheight(8)
        fig.set_figwidth(15)
        plt.savefig("results_zoom_compare_" + str(ranges[0]) +"-" + str(ranges[-1]) + ".pdf")
        #plt.show()