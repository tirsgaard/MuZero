import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from time import time
from agents import UCB1_agent, UCB1_tuned_agent, UCB1_agent_tree, Bays_agent_vector_UCT2, Bays_agent_Gauss_beta
from simulations import run_simulation, run_simulation_beta, tree_simulation, tree_thread, explore_tree, explore_parallel_tree
from scipy.optimize import nnls, minimize


if __name__ == '__main__':
    np.random.seed(2)
    n_threads = 14  # To account for virtual cores
    N_rep = n_threads*50  # Number of repeat simulations over parallel workers for reducing variance
    n_sim = 5  # Number of in-thread simulations
    N_data_points = 256  # Number of steps to go

    c_range = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    n_c_points = len(c_range)

    noise = 0.00000001
    agent_opt1 = [2, True, "no_context", 1, noise]
    agent_opt2 = [2, "UCB1", 1, noise]

    option_list = [agent_opt1]
    labels = ["UCT Bayes"]
    agents = [Bays_agent_Gauss_beta]

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
        test = explore_parallel_tree(N_data_points, agents, c_range[i], option_list)
        results = p.starmap(explore_parallel_tree, iterable=[(N_data_points, agents, c_range[i], option_list)] * N_rep)
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

    def func(theta, y_vals, return_val=False):
        c0 = theta[0]
        c1 = theta[1]
        start = theta[2]
        expon = theta[3]
        x_train = np.array(np.arange(1, y_vals.shape[0] + 1)) + start
        W_train = np.stack([c0*np.ones(x_train.shape)] + [c1 / (x_train**expon)]).T
        y_hat = W_train.sum(axis=1)
        sq_loss = np.sum((np.log(y_vals)-np.log(y_hat))**2)
        if return_val:
            return y_hat
        else:
            return sq_loss

    def get_sample_effic(ref, new, ranges):
        w_ref = minimize(func, [0, 160, 2000, 1], args=(ref,), bounds=[(-1000, 1000), (-10, 10000), (-10, 10000), (0, 5)], tol=1e-10,
                     method="Nelder-Mead", options={'maxiter': 10000})["x"]
        y_ref = func(w_ref, ref, return_val=True)

        w_new = minimize(func, [0, 160, 2000, 1], args=(new,), bounds=[(-1000, 1000), (-10, 10000), (-10, 10000), (0, 5)], tol=1e-10,
                 method="Nelder-Mead", options={'maxiter': 10000})["x"]
        y_new = func(w_new, new, return_val=True)

        n_points = new.shape[0]
        best_val = y_new[ranges]
        index = np.argmax(y_ref <= best_val)  # First time value crosses
        # Case where the new is actually more efficient
        if np.sum(y_ref <= best_val) == 0:
            best_val = y_ref[ranges]
            index = np.argmax(y_new <= best_val)
            effic = n_points / (index + 1)
        else:
            effic = (index+1) / n_points

        return effic

    # Time for plots
    def fitting_plot(ax, c_array, title, log_y=False):
        vals = c_array.mean(axis=1).squeeze()

        for i in range(n_c_points):
            w_ref = minimize(func, [0, 1, 10, 0.3], args=(vals[i, :],), bounds=[(-1000, 1000), (0, 10000), (0, 10000), (0, 1)], tol=1e-10,
                     method="Nelder-Mead", options={'maxiter': 10000})["x"]
            y_ref = func(w_ref, vals[i], return_val=True)
            ax.plot(y_ref, label=str(c_range[i]), color='C' + str(i), linestyle='dashed')
        ax.legend(fontsize='xx-small')
        ax.set_title(title)
        ax.set_xlabel("N parallel explorations")
        #plt.xticks()
        if log_y:
            ax.set_yscale("log")
        fig.set_figheight(8)
        fig.set_figwidth(15)


    def sample_effic_plot(ax, c_array, title, log_y=False):
        vals = c_array.mean(axis=1)

        for i in range(n_agents):
            effect = np.zeros((n_c_points,))
            for j in range(n_c_points):
                effect[j] = get_sample_effic(vals[0, :, i], vals[j, :, i], -1)

            label_name = labels[i]
            ax.plot(c_range, effect, label=label_name, color='C' + str(i))
        ax.legend(fontsize='xx-small')
        ax.set_title(title)
        ax.set_xlabel("N parallel explorations")
        #plt.xticks()
        if log_y:
            ax.set_yscale("log")
        ax.set_xscale("log", basex=2)
        # Set x-range values
        plt.xticks(c_range, c_range)
        fig.set_figheight(8)
        fig.set_figwidth(15)


    def speed_plot(ax, c_array, title, log_y=False):
        vals = c_array.mean(axis=1)

        for i in range(n_agents):
            effect = np.zeros((n_c_points,))
            for j in range(n_c_points):
                effect[j] = get_sample_effic(vals[0, :, i], vals[j, :, i], -1)

            label_name = labels[i]
            ax.plot(c_range, effect*np.array(c_range), label=label_name, color='C' + str(i))
        ax.legend(fontsize='xx-small')
        ax.plot(c_range, c_range, label="Ideal Speedup", color='C0', linestyle='dashed')
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("N parallel explorations")
        #plt.xticks()
        if log_y:
            ax.set_yscale("log", basey=2)
        ax.set_xscale("log", basex=2)
        # Set x-range values
        plt.xticks(c_range, c_range)
        plt.yticks(c_range, c_range)
        fig.set_figheight(8)
        fig.set_figwidth(15)


    #

    fig, ax = plt.subplots(1, 1)
    # fig.suptitle("Comparison at step " + str(ranges[0]) + "-" + str(ranges[-1]))
    ax.plot(c_regret_second_arrays.mean(axis=1).squeeze().transpose())
    fitting_plot(ax, c_regret_second_arrays, "Regret top-level decision", log_y=True)

    plt.savefig("parallel_regret.pdf")
    fig.show()

    # Compare different levels for each agent
    fig, ax = plt.subplots(1, 1)
    #fig.suptitle("Comparison at step " + str(ranges[0]) + "-" + str(ranges[-1]))
    sample_effic_plot(ax, c_regret_second_arrays, "Sample Efficiency Regret top-level decision", log_y=False)
    plt.savefig("parallel_compare_effic.pdf")

    fig, ax = plt.subplots(1, 1)
    speed_plot(ax, c_regret_second_arrays, "Speed Regret top-level decision", log_y=True)
    plt.savefig("parallel_compare_speed.pdf")
    plt.show()



