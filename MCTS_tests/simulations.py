import numpy as np
from agents import UCB1_agent, max_dist
from scipy.stats import bernoulli, beta, uniform
def tree_thread(N_data_points, agents, options):

    seed = np.random.randint(0, 10**9)
    true_mu = 0.
    true_sigma = 0.1
    min_max_tree = False
    N_bandits = 5  # Number of bandits pr. leaf
    n_agents = len(agents)

    tree_shape = (N_bandits, N_bandits, N_bandits, N_bandits)
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

    # mean reward from best bandit
    best_bandits = []
    best_bandit = bandit_mu
    for i in reversed(range(1, depth)):
        if (i % 2 == 0) or (not min_max_tree):
            best_bandits.append(best_bandit.argmax())
            best_bandit = best_bandit.max(axis=i)
        else:
            best_bandits.append(best_bandit.argmin())
            best_bandit = best_bandit.min(axis=i)

    best_reward = best_bandit.max()
    best_bandit_second = best_bandit
    best_bandit_top = best_bandit.argmax()

    #  Initialize array and fill with agent
    for j in range(len(agents)):
        np.random.seed(seed)
        agent_list = []
        for d in range(depth):
            opt = [N_bandits] + options[j] + [1, min_max_tree & (d % 2 == 1)]
            agent_shape = tree_shape[0:d]
            agent_array = np.empty((int(np.prod(agent_shape)),), dtype="object")
            # Fill array with agents
            for i in range(np.prod(agent_array.shape)):
                agent_array[i] = agents[j](*opt)  # Give arguments to agent for initialisation
            agent_array = agent_array.reshape((1, ) + agent_shape)
            agent_list.append(agent_array)

        agent_list = agent_list

        # Backup distribution from children
        for d in reversed(range(depth-1)):
            parent_array = agent_list[d]
            child_array = agent_list[d+1]
            for idx, parrent in np.ndenumerate(parent_array):
                for k in range(N_bandits):
                    backup_info = child_array[idx + (k,)].get_dist()
                    parrent.initialize_dist(k, backup_info)

        # Begin simulation
        for i in range(N_data_points):
            # Generate all actions for depths
            a_list = [[0]]
            for d in range(depth):
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
            top_action_bias[i, j] = agent_list[0][0].Q.max(axis=0) - bandit_mu[a_list[1:]]
            reward[i, j] = r

    return optimal_top_action, cum_regret, reward, greedy_array, regret_second_array, top_action_bias

def explore_parallel_noise_tree(N_data_points, agents, n_parallel, C, c_change, options):
    seed = np.random.randint(0, 10**9)
    min_max_tree = False
    N_bandits = 5 # Number of bandits pr. leaf
    n_agents = len(agents)
    n_vl = 3
    epsilon = 10**-4
    C = C  # 1000
    c_change = c_change  # 0.5
    noise = True

    tree_shape = (N_bandits, )
    depth = len(tree_shape)

    ##### Construct tree
    # Construct leafs
    alphas = np.full(tree_shape, 0.5)
    betas = np.full(tree_shape, 0.5)
    child_array = np.empty(tree_shape, dtype="object")
    for idx, parrent in np.ndenumerate(child_array):
        child_array[idx] = [alphas[idx], betas[idx]]

    # Construct tree of values
    dist_tree = [child_array]
    for d in reversed(range(1, depth)):
        dist_array = np.empty(tree_shape[:d], dtype="object")
        for idx, parrent in np.ndenumerate(dist_array):
            children = dist_tree[-1][idx]
            # Compute max distribution by iterating over children
            a, b = children[0]
            mu = a / (a + b)
            var = a * b / ((a + b) ** 2 * (a + b + 1))
            for a, b in children[1:]:
                mus = a / (a + b)
                vars = a * b / ((a + b) ** 2 * (a + b + 1))
                mu, sigma = max_dist(mu, mus, var, vars)
            alpha_p = -mu * (mu * mu - mu + sigma) / sigma
            beta_p = (mu * mu - mu + sigma) * (mu - 1) / sigma
            alpha_p = np.clip(alpha_p, epsilon, None)
            beta_p = np.clip(beta_p, epsilon, None)

            dist_array[idx] = [alpha_p, beta_p]

        dist_tree.append(dist_array)
    dist_tree.reverse()

    # Now sample from dist tree to get bernoulli parameter
    mu_tree = []
    for d in reversed(range(depth)):
        p_array = np.empty(tree_shape[(depth-d-1):], dtype=np.float32)
        for idx, parrent in np.ndenumerate(p_array):
            if (d==(depth-1)):
                # Sample from leaves
                a, b = dist_tree[d][idx]
                p = beta.rvs(a, b)
                p_array[idx] = p
            else:
                # Pass values up the tree
                p_array[idx] = np.max(mu_tree[-1][idx])


        mu_tree.append(p_array)
    mu_tree.reverse()

    # Construct context tree
    # Based on sampled value of p
    context_tree = []
    for d in reversed(range(depth)):
        context_array = np.empty(tree_shape[(depth-d-1):], dtype="object")
        for idx, parrent in np.ndenumerate(context_array):
            # Compute max distribution
            a, b = dist_tree[d][idx]
            p = mu_tree[d][idx]
            # Calculate contracted dist
            a_cont = p * C
            b_cont = C - p * C
            if noise:
                p_new = beta.rvs(a_cont, b_cont)
                p_new = c_change * p_new + (1-c_change)*p
                a_cont = p_new * C
                b_cont = C - p_new * C

            context_array[idx] = [a_cont, b_cont]
        context_tree.append(context_array)
    context_tree.reverse()

    cum_regret = np.zeros((N_data_points, n_agents))
    greedy_array = np.zeros((N_data_points, n_agents))
    regret_second_array = np.zeros((N_data_points, n_agents))
    optimal_action = np.zeros((N_data_points, n_agents))
    optimal_top_action = np.zeros((N_data_points, n_agents))
    reward = np.zeros((N_data_points, n_agents))
    top_action_bias = np.zeros((N_data_points, n_agents))

    # mean reward from best bandit
    best_bandit_idx = np.unravel_index(mu_tree[-1].argmax(), mu_tree[-1].shape)
    best_bandit = mu_tree[-1]
    top_level_rewards = np.empty((N_bandits,))
    for i in range(N_bandits):
        top_level_rewards[i] = np.max(mu_tree[-1][i])

    best_reward = mu_tree[-1].max()
    best_bandit_top = best_bandit[0]

    #  Initialize array and fill with agent
    for j in range(len(agents)):
        np.random.seed(seed)
        agent_list = []
        for d in range(depth):
            opt = [N_bandits] + options[j] + [1, min_max_tree & (d % 2 == 1)]
            agent_shape = tree_shape[0:d]
            agent_array = np.empty((int(np.prod(agent_shape)),), dtype="object")
            # Fill array with agents
            for i in range(np.prod(agent_array.shape)):
                agent_array[i] = agents[j](*opt)  # Give arguments to agent for initialisation
            agent_array = agent_array.reshape((1, ) + agent_shape)
            agent_list.append(agent_array)


        # Set context values
        for d in range(depth):
            parent_array = agent_list[d][0]
            for idx, parrent in np.ndenumerate(parent_array):
                alphas = np.zeros((N_bandits,))
                betas = np.zeros((N_bandits,))
                for k in range(N_bandits):
                    # Add values to policies
                    alphas[k], betas[k] = context_tree[d][idx + (k,)]
                # Backup
                parrent.set_context(alphas, betas, True)

        # Backup distribution from children
        for d in reversed(range(depth-1)):
            parent_array = agent_list[d]
            child_array = agent_list[d+1]
            for idx, parrent in np.ndenumerate(parent_array):
                for k in range(N_bandits):
                    backup_info = child_array[idx + (k,)].get_dist()
                    parrent.initialize_dist(k, backup_info)

        ####### Begin simulation
        i = 0
        while i < N_data_points:
            n_jobs = min((N_data_points-i), n_parallel)
            a_list_array = np.zeros((n_parallel,), dtype=object)
            agents_selected_array = np.zeros((n_parallel,), dtype=object)
            depth_array = np.zeros((n_parallel,), dtype=np.uint64)
            r_array = np.zeros((n_parallel, ), dtype=np.float32)
            succes_array = np.ones((n_parallel,), dtype=np.bool)

            for k in range(n_jobs):
                # Generate all actions for depths
                a_list = [[0]]
                agents_selected = []
                for d in range(depth):
                    agent = agent_list[d][a_list][0]
                    action = agent.act()
                    a_list.append(action)
                    agents_selected.append(agent)

                    if not agent.explored:
                        #  Case where the same node is about to be explored in parallel
                        if agent.i > 0:
                            succes_array[k] = False
                        break
                # Add virtual loss
                extra_info = None
                for agent, action in zip(reversed(agents_selected), reversed(a_list)):
                    extra_info = agent.backup(action[0], 0, extra_info, vl=True)

                a_list_array[k] = a_list
                agents_selected_array[k] = agents_selected
                depth_array[k] = d

                if not succes_array[k]:
                    # Stop doing more work as this will be explored by another thread
                    continue

                # Get payout
                selected_bandit_p = mu_tree[d][a_list[1:]]
                r = bernoulli.rvs(selected_bandit_p)
                ### Save results
                r_array[k] = r

            #### Backup
            # Remove Virtual loss
            for k in range(n_jobs):
                a_list = a_list_array[k]
                agents_selected = agents_selected_array[k]
                # Backup result
                extra_info = None
                for agent, action in zip(reversed(agents_selected), reversed(a_list)):
                    extra_info = agent.undo_vl(action[0], 0, extra_info)

            # Backup result
            for k in range(n_jobs):
                if not succes_array[k]:
                    continue  # Skip if already explored
                # Get data
                a_list = a_list_array[k]
                agents_selected = agents_selected_array[k]
                depth_obtained = int(depth_array[k])
                r = r_array[k]
                # Backup actual values
                extra_info = None
                for agent, action in zip(reversed(agents_selected), reversed(a_list)):
                    extra_info = agent.backup(action[0], r, extra_info)

                ##### Get statistics
                # Average regret
                cum_regret[i, j] = cum_regret[i - 1, j] + best_reward - mu_tree[depth_obtained][a_list[1:]]
                # Greedy error
                greedy_action = np.zeros((depth, ), dtype=np.int64)
                a_list = [[0]]
                for d in range(depth):
                    greedy_action[d] = int(agent_list[d][a_list][0].get_greedy_action())
                    a_list.append([greedy_action[d]])

                greedy_array[i, j] = mu_tree[-1][a_list[1:]]
                opt_action = greedy_action == best_bandit_idx
                optimal_action[i, j] = np.all(opt_action, axis=0)
                optimal_top_action[i, j] = 1 - (agent_list[0][0].Q.argmax(axis=0) == best_bandit_idx[0])
                regret_second_array[i, j] = best_reward - top_level_rewards[a_list[1]]
                top_action_bias[i, j] = agent_list[0][0].Q.max(axis=0) - top_level_rewards[a_list[1]]
                reward[i, j] = r

                i += 1

    return optimal_top_action, cum_regret, reward, greedy_array, regret_second_array, top_action_bias


def explore_parallel_tree(N_data_points, agents, n_parallel, options):
    seed = np.random.randint(0, 10**9)
    min_max_tree = False
    N_bandits = 5  # Number of bandits pr. leaf
    n_agents = len(agents)
    n_vl = 3

    tree_shape = (N_bandits, N_bandits)
    depth = len(tree_shape)

    ##### Construct tree
    mu_tree = []
    # Construct leafs
    leaf_mu = beta.rvs(2, 5, size=tree_shape)#np.random.rand(*tree_shape)
    child_array = leaf_mu
    mu_tree.append(leaf_mu)
    for d in reversed(range(depth)):
        mu_array = np.empty(tree_shape[:d])
        if (d % 2 == 0) or (not min_max_tree):
            for idx, parrent in np.ndenumerate(mu_array):
                mu_array[idx] = np.max(child_array[idx])
        else:
            for idx, parrent in np.ndenumerate(mu_array):
                mu_array[idx] = np.min(child_array[idx])
        mu_tree.append(mu_array)
        child_array = child_array

    mu_tree.reverse()

    cum_regret = np.zeros((N_data_points, n_agents))
    greedy_array = np.zeros((N_data_points, n_agents))
    regret_second_array = np.zeros((N_data_points, n_agents))
    optimal_action = np.zeros((N_data_points, n_agents))
    optimal_top_action = np.zeros((N_data_points, n_agents))
    reward = np.zeros((N_data_points, n_agents))
    top_action_bias = np.zeros((N_data_points, n_agents))

    # mean reward from best bandit
    best_bandits = []
    best_bandit = leaf_mu
    for i in reversed(range(1, depth)):
        if (i % 2 == 0) or (not min_max_tree):
            best_bandits.append(best_bandit.argmax())
            best_bandit = best_bandit.max(axis=i)
        else:
            best_bandits.append(best_bandit.argmin())
            best_bandit = best_bandit.min(axis=i)

    best_reward = best_bandit.max()
    best_bandit_second = best_bandit
    best_bandit_top = best_bandit.argmax()

    #  Initialize array and fill with agent
    for j in range(len(agents)):
        np.random.seed(seed)
        agent_list = []
        for d in range(depth):
            opt = [N_bandits] + options[j] + [1, min_max_tree & (d % 2 == 1)]
            agent_shape = tree_shape[0:d]
            agent_array = np.empty((int(np.prod(agent_shape)),), dtype="object")
            # Fill array with agents
            for i in range(np.prod(agent_array.shape)):
                agent_array[i] = agents[j](*opt)  # Give arguments to agent for initialisation
            agent_array = agent_array.reshape((1, ) + agent_shape)
            agent_list.append(agent_array)


        # Set context values
        for d in reversed(range(depth)):
            parent_array = agent_list[d][0]
            for idx, parrent in np.ndenumerate(parent_array):
                P_vals = np.zeros((N_bandits, 1))
                for k in range(N_bandits):
                    # Add values to policies
                    P_vals[k] = mu_tree[d + 1][idx + (k,)]
                # Backup
                parrent.set_context(P_vals, d == (depth-1))

        # Backup distribution from children and add context
        for d in reversed(range(depth-1)):
            parent_array = agent_list[d]
            child_array = agent_list[d+1]
            for idx, parrent in np.ndenumerate(parent_array):
                for k in range(N_bandits):
                    backup_info = child_array[idx + (k,)].get_dist()
                    parrent.initialize_dist(k, backup_info)

        ####### Begin simulation
        i = 0
        while i < N_data_points:
            n_jobs = min((N_data_points-i), n_parallel)
            a_list_array = np.zeros((n_parallel,), dtype=object)
            agents_selected_array = np.zeros((n_parallel,), dtype=object)
            depth_array = np.zeros((n_parallel,), dtype=np.uint64)
            r_array = np.zeros((n_parallel, ), dtype=np.float32)
            succes_array = np.ones((n_parallel,), dtype=np.bool)

            for k in range(n_jobs):
                # Generate all actions for depths
                a_list = [[0]]
                agents_selected = []
                for d in range(depth):
                    agent = agent_list[d][a_list][0]
                    action = agent.act()
                    a_list.append(action)
                    agents_selected.append(agent)

                    if not agent.explored:
                        #  Case where the same node is about to be explored in parallel
                        if agent.i > 0:
                            succes_array[k] = False
                        break
                # Add virtual loss
                extra_info = None
                for agent, action in zip(reversed(agents_selected), reversed(a_list)):
                    extra_info = agent.backup(action[0], 0, extra_info, vl=True)

                a_list_array[k] = a_list
                agents_selected_array[k] = agents_selected
                depth_array[k] = d

                if not succes_array[k]:
                    # Stop doing more work as this will be explored by another thread
                    continue

                # Get payout
                selected_bandit_p = mu_tree[d+1][a_list[1:]]
                r = bernoulli.rvs(selected_bandit_p)
                ### Save results
                r_array[k] = r

            #### Backup
            # Remove Virtual loss
            for k in range(n_jobs):
                a_list = a_list_array[k]
                agents_selected = agents_selected_array[k]
                # Backup result
                extra_info = None
                for agent, action in zip(reversed(agents_selected), reversed(a_list)):
                    extra_info = agent.undo_vl(action[0], 0, extra_info)

            # Backup result
            for k in range(n_jobs):
                if not succes_array[k]:
                    continue  # Skip if already explored
                # Get data
                a_list = a_list_array[k]
                agents_selected = agents_selected_array[k]
                depth_obtained = int(depth_array[k])
                r = r_array[k]
                # Backup actual values
                extra_info = None
                for agent, action in zip(reversed(agents_selected), reversed(a_list)):
                    extra_info = agent.backup(action[0], r, extra_info)

                ##### Get statistics
                # Average regret
                cum_regret[i, j] = cum_regret[i - 1, j] + best_reward - mu_tree[depth_obtained+1][a_list[1:]]
                # Greedy error
                greedy_action = np.zeros((depth, ), dtype=np.int64)
                a_list = [[0]]
                for d in range(depth):
                    greedy_action[d] = int(agent_list[d][a_list][0].get_greedy_action())
                    a_list.append([greedy_action[d]])

                greedy_array[i, j] = mu_tree[-1][a_list[1:]]
                opt_action = greedy_action == best_bandits
                optimal_action[i, j] = np.all(opt_action, axis=0)
                optimal_top_action[i, j] = 1 - (agent_list[0][0].Q.argmax(axis=0) == best_bandit_top)
                regret_second_array[i, j] = best_reward - best_bandit_second[a_list[1]]
                top_action_bias[i, j] = agent_list[0][0].Q.max(axis=0) - mu_tree[1][a_list[1]]
                reward[i, j] = r

                i += 1

    return optimal_top_action, cum_regret, reward, greedy_array, regret_second_array, top_action_bias

def explore_tree(N_data_points, agents, options):
    seed = np.random.randint(0, 10**9)
    min_max_tree = False
    N_bandits = 5  # Number of bandits pr. leaf
    n_agents = len(agents)

    tree_shape = (N_bandits, N_bandits, N_bandits, N_bandits)
    depth = len(tree_shape)

    ##### Construct tree
    mu_tree = []
    # Construct leafs
    leaf_mu = beta.rvs(2, 5, size=tree_shape)#np.random.rand(*tree_shape)
    child_array = leaf_mu
    mu_tree.append(leaf_mu)
    for d in reversed(range(depth)):
        mu_array = np.empty(tree_shape[:d])
        if (d % 2 == 0) or (not min_max_tree):
            for idx, parrent in np.ndenumerate(mu_array):
                mu_array[idx] = np.max(child_array[idx])
        else:
            for idx, parrent in np.ndenumerate(mu_array):
                mu_array[idx] = np.min(child_array[idx])
        mu_tree.append(mu_array)
        child_array = child_array

    mu_tree.reverse()

    cum_regret = np.zeros((N_data_points, n_agents))
    greedy_array = np.zeros((N_data_points, n_agents))
    regret_second_array = np.zeros((N_data_points, n_agents))
    optimal_action = np.zeros((N_data_points, n_agents))
    optimal_top_action = np.zeros((N_data_points, n_agents))
    reward = np.zeros((N_data_points, n_agents))
    top_action_bias = np.zeros((N_data_points, n_agents))

    # mean reward from best bandit
    best_bandits = []
    best_bandit = leaf_mu
    for i in reversed(range(1, depth)):
        if (i % 2 == 0) or (not min_max_tree):
            best_bandits.append(best_bandit.argmax())
            best_bandit = best_bandit.max(axis=i)
        else:
            best_bandits.append(best_bandit.argmin())
            best_bandit = best_bandit.min(axis=i)

    best_reward = best_bandit.max()
    best_bandit_second = best_bandit
    best_bandit_top = best_bandit.argmax()

    #  Initialize array and fill with agent
    for j in range(len(agents)):
        np.random.seed(seed)
        agent_list = []
        for d in range(depth):
            opt = [N_bandits] + options[j] + [1, min_max_tree & (d % 2 == 1)]
            agent_shape = tree_shape[0:d]
            agent_array = np.empty((int(np.prod(agent_shape)),), dtype="object")
            # Fill array with agents
            for i in range(np.prod(agent_array.shape)):
                agent_array[i] = agents[j](*opt)  # Give arguments to agent for initialisation
            agent_array = agent_array.reshape((1, ) + agent_shape)
            agent_list.append(agent_array)


        # Set context values
        for d in reversed(range(depth)):
            parent_array = agent_list[d][0]
            for idx, parrent in np.ndenumerate(parent_array):
                P_vals = np.zeros((N_bandits, 1))
                for k in range(N_bandits):
                    # Add values to policies
                    P_vals[k] = mu_tree[d + 1][idx + (k,)]
                # Backup
                parrent.set_context(P_vals, True)

        # Backup distribution from children and add context
        for d in reversed(range(depth-1)):
            parent_array = agent_list[d]
            child_array = agent_list[d+1]
            for idx, parrent in np.ndenumerate(parent_array):
                for k in range(N_bandits):
                    backup_info = child_array[idx + (k,)].get_dist()
                    parrent.initialize_dist(k, backup_info)

        ####### Begin simulation
        for i in range(N_data_points):
            ####
            # Generate all actions for depths
            a_list = [[0]]
            agents_selected = []
            for d in range(depth):
                agent = agent_list[d][a_list][0]
                action = agent.act()
                a_list.append(action)
                agents_selected.append(agent)
                if agent.i == 0:
                    break
            depth_obtained = d
            # Get payout
            selected_bandit_p = mu_tree[depth_obtained+1][a_list[1:]]
            r = bernoulli.rvs(selected_bandit_p)  # np.random.normal(bandit_mu[a_list[1:]], bandit_sigma[a_list[1:]])[0]
            # Backup result
            extra_info = None

            #### Backup
            for agent, action in zip(reversed(agents_selected), reversed(a_list)): # Note the obtained depth from last loop is used here
                extra_info = agent.backup(action[0], r, extra_info)

            ##### Get statistics
            # Average regret
            cum_regret[i, j] = cum_regret[i - 1, j] + best_reward - mu_tree[depth_obtained+1][a_list[1:]]
            # Greedy error
            greedy_action = np.zeros((depth, ), dtype=np.int64)
            a_list = [[0]]
            for d in range(depth):
                greedy_action[d] = int(agent_list[d][a_list][0].get_greedy_action())
                a_list.append([greedy_action[d]])

            greedy_array[i, j] = mu_tree[-1][a_list[1:]]
            opt_action = greedy_action == best_bandits
            optimal_action[i, j] = np.all(opt_action, axis=0)
            optimal_top_action[i, j] = 1 - (agent_list[0][0].Q.argmax(axis=0) == best_bandit_top)
            regret_second_array[i, j] = best_reward - best_bandit_second[a_list[1]]
            top_action_bias[i, j] = agent_list[0][0].Q.max(axis=0) - mu_tree[1][a_list[1]]
            reward[i, j] = r

    return optimal_top_action, cum_regret, reward, greedy_array, regret_second_array, top_action_bias

def tree_simulation(N_data_points, n_sim, agents):
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


def tree_simulation(N_data_points, n_sim, agents):
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