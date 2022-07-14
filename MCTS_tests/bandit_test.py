import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import time
from agents import UCB1_agent, UCB1_agent_tree, Bays_agent_vector_UCT2, Bays_agent_Gauss_beta
from simulations import run_simulation, run_simulation_beta, tree_simulation, tree_thread, explore_tree, explore_parallel_tree, explore_parallel_noise_tree
import pickle

if __name__ == '__main__':
    np.random.seed(2)
    n_threads = 16
    N_rep = n_threads*50  # Number of repeat simulations over parallel workers for reducing variance # 400*3000 is enough for UCB
    n_sim = 5  # Number of in-thread simulations
    N_data_points = 1000  # Number of steps to go
    noise = 0.000000001
    n_parallel = 1
    #agent_opt1 = [2,True, np.linspace(0, 1, 4000), "UCT1"]
    agent_opt2 = [2, True, "UCT1_extracontext", 1, noise]
    agent_opt3 = [2, "epsilon", 1, noise]
    agent_opt4 = [2, True, np.linspace(0, 1, 20), "UCT1"]

    #agent_opt3 = [2, True, np.linspace(0, 1, 4000), "thompson"]
    #agent_opt4 = [2, False, "PUCT_context", 1, noise]
    #agent_opt5 = [0.5, True, "UCT1_context", 1, noise]
    #agent_opt5 = [2, False, "thompson", 1, noise]
    #agent_opt6 = [2, False, "thompson", 1, noise]
    #agent_opt6 = [2, True, "alphaGo", 1, noise]

    option_list = [agent_opt4, agent_opt2, agent_opt3]
    labels = ["Vector UCT", "Bayes UCT", "Epsilon s = 1"]
    agents = [Bays_agent_vector_UCT2, Bays_agent_Gauss_beta, UCB1_agent]

    assert(len(option_list) == len(labels))
    assert (len(option_list) == len(agents))

    t = time()
    test = explore_parallel_noise_tree(N_data_points, agents, n_parallel, 10, 1, option_list)

    p = Pool(n_threads)
    results = p.starmap(explore_parallel_noise_tree, iterable=[(N_data_points, agents, n_parallel, 10, 1, option_list)] * N_rep)
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
    ax1.legend(fontsize='medium')

    ax1.set_title("Error on action selected")
    ax1.set_yscale("log")
    #ax1.set_ylim(0.1, 1)
    #plt.xscale("log")
    #plt.savefig("Opt_act.png")
    #plt.show()

    ax2.plot(np.mean(regrets, axis=0), label=labels)
    ax2.legend()
    ax2.set_title("Cumulative regret")
    #ax2.set_yscale("log")
    #plt.savefig("Cum_regret.png")
    #plt.show()

    ax3.plot(np.mean(regret_second, axis=0), label=labels)
    ax3.legend()
    ax3.set_title("Regret top-level decision")
    #ax3.set_ylim(0.007, 0.2)
    ax3.set_yscale("log")
    ax3.set_xlabel("Steps")
    # fit model
    y_vals = np.mean(regret_second, axis=0)
    x_train = np.array(np.arange(1, y_vals.shape[0] + 1))

    def func(theta, return_val=False):
        c0 = theta[0]
        c1 = theta[1]
        start = theta[2]
        expon = theta[3]
        x_train = np.array(np.arange(1, y_vals.shape[0] + 1)) + start
        W_train = np.stack([c0*np.ones(x_train.shape)] + [c1 / (x_train**expon)]).T
        y_hat = W_train.sum(axis=1)
        sq_loss = np.sum((np.log(y_vals[:, 0])-np.log(y_hat))**2)
        if return_val:
            return y_hat
        else:
            return sq_loss



    from scipy.optimize import nnls, minimize
    # Fit monotonic regression
    """
        def basis_matrix(inputs, orders):
        return np.stack([np.ones(inputs.shape)] + [1 / (inputs**o) for o in range(1, orders + 1)]).T
    W_train = basis_matrix(x_train, 3)
    # w = np.linalg.solve(W.T @ W, W.T @ y) # Ordinaly least squares (just for comparison)
    w = nnls(W_train, y_vals[:, 0])[0]

    y_hat = W_train @ w
    """

    w = minimize(func, [0, 160, 2000, 1], bounds=[(-10000, 10000), (0, 10000), (0, 10000), (1, 1)], tol=1e-10, method="Nelder-Mead", options={'maxiter': 10000})["x"]
    y_hat = func(w, return_val=True)
    ax3.plot(y_hat)

    #plt.xscale("log")
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
    plt.savefig("results.pdf")
    plt.show()

    # Save regret plot
    fig, ax = plt.subplots(1, 1)
    #ax.plot(np.mean(regret_second, axis=0), label=labels)
    for i in range(len(agents)):
        vals = regret_second[:, :, i].mean(axis=0)
        sd = np.sqrt( regret_second[:, :, i].var(axis=0) / N_rep )
        label_name = labels[i]
        ax.plot(vals, label=label_name, color='C' + str(i))
        ax.plot(vals + 2 * sd, '--', color='C' + str(i))
        ax.plot(vals - 2 * sd, '--', color='C' + str(i))

    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.mean(regrets, axis=0), label=labels)
    ax.legend()
    #ax.set_title("Cumulative regret")
    ax.set_xlabel("N")
    ax.set_ylabel("Cumulative Regret")
    #ax.set_yscale("log")
    plt.savefig("Cum_regret.pdf")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.mean(optimal_actions, axis=0), label=labels)
    ax.legend()
    #ax.set_title("Cumulative regret")
    ax.set_xlabel("N")
    ax.set_ylabel("Greedy choice Error")
    #ax.set_yscale("log")
    plt.savefig("Greedy_choice.pdf")
    plt.show()

    """
    #  Plot more
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.mean(regret_second, axis=0), label=labels[0])
    ax.legend()
    ax.set_ylabel("Greedy Regret Top-level")
    #ax3.set_ylim(0.007, 0.2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    # fit model
    y_vals = np.mean(regret_second, axis=0)
    x_train = np.array(np.arange(1, y_vals.shape[0] + 1))

    w = minimize(func, [0, 160, 2000, 1], bounds=[(-1000, 1000), (0, 10000), (0, 10000), (1, 1)], tol=1e-10, method="Nelder-Mead", options={'maxiter': 10000})["x"]
    y_hat = func(w, return_val=True)
    ax.plot(y_hat, label=r'$' + str(round(w[0], 5)) + r'+\frac{' + str(round(w[1], 4)) + r'}{N + ' + str(round(w[2], 3)) + r'}$')

    w = minimize(func, [0, 160, 2000, 1], bounds=[(0, 0), (0, 10000), (0, 10000), (1, 1)], tol=1e-10,
                 method="Nelder-Mead", options={'maxiter': 10000})["x"]
    y_hat = func(w, return_val=True)
    ax.plot(y_hat, label=r'$\frac{' + str(round(w[1], 4)) + r'}{N + ' + str(round(w[2], 3)) + r'}$')
    ax.legend()

    plt.show()
    with open('regret_figure_d1.pkl', 'wb') as file:
        pickle.dump(ax, file)
    fig.savefig("regret_figure.pdf")
    """
