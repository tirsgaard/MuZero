from line_profiler_pycharm import profile
import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
from time import time
import math

def Phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def phi(x):
    # Propability mass distribution of the standard normal distribution
    return 0.3989422802*math.exp(-0.5*x*x)  # Number is 1/sqrt(2*pi)

class UCB1_agent_tree:
    def __init__(self, bandit_shape, n_sim=1):
        self.depth = len(bandit_shape)
        self.Qs = []
        self.Vs = []
        self.n_obs = []
        self.total_obs = []

        number_counts = (n_sim, )
        # Store bandit statistics for each depth
        for i in range(1, len(bandit_shape) + 1):
            self.Qs.append(np.zeros(bandit_shape[-i:] + (n_sim,)) + np.float("inf") )
            self.Vs.append(np.zeros(bandit_shape[-i:] + (n_sim,)) )
            self.n_obs.append(np.zeros(bandit_shape[-i:] + (n_sim,), dtype="uint16") )
            self.total_obs.append(np.zeros(number_counts, dtype="uint16") )
            number_counts = (bandit_shape[-i], ) + number_counts
        self.c = 1.4142135623730951  #1.
        self.n_sim = n_sim

    def act(self):
        a = np.empty((self.n_sim, self.depth), dtype=np.int32)
        i = 0
        a[:, 0] = np.argmax(self.Qs[i] + self.c * np.sqrt(np.log(self.total_obs[0]) / self.n_obs[i]), axis=0)
        i = 1
        a[:, i] = np.argmax(self.Qs[i][a[:, 0], :, range(self.n_sim)] + self.c * np.sqrt(np.log(self.total_obs[1][a[:, 0], range(self.n_sim)][:, None]) / self.n_obs[i][a[:, 0], :, range(self.n_sim)]), axis=1)
        i = 2
        a[:, i] = np.argmax(self.Qs[i][a[:, 0], a[:, 1], :, range(self.n_sim)] + self.c * np.sqrt(
            np.log(self.total_obs[2][a[:, 0], a[:, 1], range(self.n_sim)][:, None]) / self.n_obs[i][a[:, 0], a[:, 1], :, range(self.n_sim)]), axis=1)
        return a

    def backup(self, a, r):
        indexes = [range(self.n_sim)]
        for i in range(self.depth):
            indexes.insert(i, a[range(self.n_sim), i])
            self.Vs[i][indexes] += r
            self.n_obs[i][indexes] += 1
            self.Qs[i][indexes] = self.Vs[i][indexes] / self.n_obs[i][indexes]
            self.total_obs[i][indexes[0:i] + [indexes[-1]]] += 1


class UCB1_agent:
    def __init__(self, N_bandits, c=2, n_sim=1):
        self.Q = np.zeros((N_bandits, n_sim)) + np.float("inf")
        self.V = np.zeros((N_bandits, n_sim))
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.c = c
        self.i = 0
        self.n_sim = n_sim

    def act(self):
        a = np.argmax(self.Q + np.sqrt(self.c*np.log(self.i)/self.n_obs), axis=0)
        return a

    def act_min(self):
        a = np.argmax((1-self.Q) + np.sqrt(self.c*np.log(self.i)/self.n_obs), axis=0)
        return a

    def backup(self, a, r, extra_info=None):
        self.V[a, range(self.n_sim)] += r
        self.n_obs[a, range(self.n_sim)] += 1
        self.Q[a, range(self.n_sim)] = self.V[a, range(self.n_sim)] / self.n_obs[a, range(self.n_sim)]
        self.i += 1
        return extra_info

class UCB1_tuned_agent:
    def __init__(self, N_bandits, n_sim = 1, upper_variance = 2, known_variance = True):
        self.Q = np.zeros((N_bandits, n_sim)) + np.float("inf")
        self.V = np.zeros((N_bandits, n_sim))
        self.V2 = np.zeros((N_bandits, n_sim)) # This is sum of squared reward
        self.UV = np.zeros((N_bandits, n_sim))
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.upper_variance = np.zeros((N_bandits, n_sim)) + upper_variance
        self.known_variance = known_variance
        self.n_sim = n_sim
        self.c = 2
        self.i = 1

    def act(self):
        if self.known_variance:
            U_var = self.upper_variance
        else:
            U_var = np.minimum(self.upper_variance, self.UV+np.sqrt(2*np.log(self.i)/self.n_obs))
        a = np.argmax(self.Q + self.c*np.sqrt( np.log(self.i)/self.n_obs * U_var), axis=0)
        return a

    def backup(self, a, r):
        self.V[a, range(self.n_sim)] += r
        self.V2[a, range(self.n_sim)] += r*r
        self.n_obs[a, range(self.n_sim)] += 1
        self.Q[a, range(self.n_sim)] = self.V[a, range(self.n_sim)] / self.n_obs[a, range(self.n_sim)]
        self.UV[a, range(self.n_sim)] = (1/self.n_obs[a, range(self.n_sim)])*self.V2[a, range(self.n_sim)] - self.Q[a, range(self.n_sim)]*self.Q[a, range(self.n_sim)]
        self.i += 1

class UCB1_normal_agent:
    def __init__(self, N_bandits, n_sim = 1):
        self.Q = np.zeros((N_bandits, n_sim)) + np.float("inf")
        self.V = np.zeros((N_bandits, n_sim))
        self.q = np.zeros((N_bandits, n_sim))
        self.n_obs = np.zeros((N_bandits,  n_sim), dtype="uint16")
        self.i = 1
        self.n_sim = n_sim

    def act(self):
        min_play = np.min(self.n_obs, axis=0)

        temp = (self.q - self.n_obs * (self.Q ** 2)) * np.log(self.i - 1) / ((self.n_obs - 1) * self.n_obs)
        temp1 = np.argmax(self.Q + (16 * temp) ** 0.5, axis=0)
        temp2 = np.argmin(self.n_obs, axis=0)
        cond = min_play<np.ceil(8*np.log(self.i))
        a = np.where(cond, temp2, temp1)
        return a

    def backup(self, a, r):
        self.q[a, range(self.n_sim)] += r * r
        self.V[a, range(self.n_sim)] += r
        self.n_obs[a, range(self.n_sim)] += 1
        self.Q[a, range(self.n_sim)] = self.V[a, range(self.n_sim)] / self.n_obs[a, range(self.n_sim)]
        self.i += 1

class UCB2_agent:
    def __init__(self, N_bandits, c=2, n_sim = 1):
        self.Q = np.zeros((N_bandits, n_sim)) + np.float("inf")
        self.V = np.zeros((N_bandits, n_sim))
        self.r = np.zeros((N_bandits, n_sim))
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.alpha = 0.1
        self.i = 1
        self.remaining_play = np.zeros((n_sim,), dtype="int16")
        self.to_play        = np.zeros((n_sim,), dtype="int16")
        self.n_sim = n_sim
        self.indexer = range(self.n_sim) # This is just for sanity

    def tau(self, r):
        return np.ceil((1+self.alpha)**r)

    def act(self):

        self.remaining_play -= 1

        not_reset = self.remaining_play>0
        reset = np.logical_not(not_reset)

        # Update a values after epoch for the last chosen action
        self.r[self.to_play, self.indexer] += reset
        tau_cur = self.tau(self.r)
        self.a = np.sqrt((1 + self.alpha) * np.log(np.exp(1) * self.i / tau_cur) / (2 * tau_cur))
        a = np.argmax(self.Q + self.a, axis = 0)
        self.to_play[reset] = a[reset]
        self.remaining_play[reset] = (self.tau(self.r[a, self.indexer]+1) - self.tau(self.r[a, self.indexer]))[reset]

        return self.to_play

    def backup(self, a, r):
        self.V[a, self.indexer] += r
        self.n_obs[a, self.indexer] += 1
        self.Q[a, self.indexer] = self.V[a, self.indexer] / self.n_obs[a, self.indexer]
        self.i += 1


class Bays2_agent:
    def __init__(self, N_bandits, prior_mean, prior_var, n_sim = 1):
        self.Q = np.zeros((N_bandits, n_sim)) + np.float("inf")
        self.V = np.zeros((N_bandits, n_sim))
        self.sigma1 = np.zeros((N_bandits, n_sim))
        self.sigma2 = np.zeros((N_bandits, n_sim))
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.i = 1
        self.n_sim = n_sim
        self.indexer = range(self.n_sim) # This is just for sanity

    def act(self):
        a = np.argmax(self.mu + np.sqrt(2*np.log(self.i)*self.sigma), axis = 0)
        return a

    def backup(self, a, r):
        self.V[a, self.indexer] += r
        self.n_obs[a, self.indexer] += 1
        self.Q[a, self.indexer] = self.V[a, self.indexer] / self.n_obs[a, self.indexer]
        sigmam = self.sigma1*self.sigma1 + self.sigma2*self.sigma2# No correlation term included
        alpha = (self.mu1-self.mu2)/sigmam
        Phi_alpha = norm.cdf(alpha)
        phi_alpha = norm.pdf(alpha)
        F1 = alpha*Phi_alpha+phi_alpha
        F2 = alpha*alpha*Phi_alpha*(1-Phi_alpha)+(1-2*Phi_alpha)*alpha*phi_alpha-phi_alpha*phi_alpha

        self.mu = self.mu2+sigmam*F1
        self.sigma = self.sigma2+(self.sigma1-self.sigma2)*Phi_alpha+sigmam*F2

        self.i += 1


class Bayes_agent:
    def __init__(self, N_bandits, mu_prior, sigma_prior, N_data_points):
        ## Reset statistics
        self.mu_hat = np.zeros(N_bandits)
        self.n_obs = np.zeros(N_bandits, dtype="uint16")
        self.sigma_obs = np.zeros(N_bandits) + float("inf")
        self.mu_obs = np.zeros(N_bandits)
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.sigma_post = sigma_prior.copy()
        self.mu_post = mu_prior.copy()
        self.returns_cum = np.zeros(N_bandits)
        self.obs = np.zeros((N_bandits, N_data_points))

    def act(self):
        mu_diff = self.mu_post[1] - self.mu_post[0]
        prop_1 = Phi(mu_diff / np.sqrt(self.sigma_post[0] ** 2 + self.sigma_post[1] ** 2))  # Chance bandit 2 is best
        prop_0 = 1 - prop_1  # Chance bandit 1 is best
        a = np.random.choice(2, p=np.array([prop_0, prop_1]))
        return a

    def backup(self, a, r):
        self.n_obs[a] += 1
        self.obs[a, self.n_obs[a]] = r
        self.returns_cum[a] += r

        ### Update sample statistics
        data = self.obs[a, 0:self.n_obs[a]]
        self.sigma_obs[a] = np.var(data, ddof=1) ** 0.5
        # Take care of first sampling
        if self.n_obs[a] == 1:
            self.sigma_obs[a] = float("inf")
        self.mu_obs[a] = np.mean(data)

        ### Update posterior (This would easier if done via precision instead of variance)
        self.sigma_post[a] = 1 / (1 / (self.sigma_prior[a] ** 2) + self.n_obs[a] / (self.sigma_obs[a] ** 2))
        self.mu_post[a] = self.sigma_post[a] * (self.mu_prior[a] / self.sigma_prior[a] ** 2 + self.returns_cum[a] / self.sigma_obs[a] ** 2)

class Bays_agent_Gauss_UCT2:
    def __init__(self, N_bandits, n_sim=1):
        self.Q = np.zeros((N_bandits, n_sim)) + np.float("inf")
        self.V = np.zeros((N_bandits, n_sim))  # Store sum of returns
        self.W = np.ones((N_bandits, n_sim))  # Store sum of squared returns
        self.Sigmas = np.ones((N_bandits, n_sim))  # These are sigma^2
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.N_bandits = N_bandits
        self.c = 2
        self.i = 0
        self.n_sim = n_sim

    def combine(self, mu1, mu2, sigma1, sigma2):
        # Note sigma needs to be squared
        rho = 0
        sigma_m = sigma1+sigma2-2*rho*sigma1*sigma2
        alpha = (mu1-mu2)/sigma_m

        Phi_alpha = Phi(alpha)  # Store to avoid
        phi_alpha = phi(alpha)
        F1 = alpha*Phi_alpha + phi_alpha
        F2 = alpha*alpha*Phi_alpha*(1-Phi_alpha) + (1-2*Phi_alpha)*alpha*phi_alpha - phi_alpha*phi_alpha
        mu = mu2 + sigma_m*F1
        sigma = sigma2 + (sigma1-sigma2)*Phi_alpha + sigma_m * F2
        return mu, sigma

    def calc_max_dist(self):
        mu = self.Q[0][0]
        sigma = self.Sigmas[0][0]
        for i in range(1, self.N_bandits):
            mu, sigma = self.combine(mu, self.Q[i][0], sigma, self.Sigmas[i][0])
        return mu, sigma

    def act(self):
        a = np.argmax(self.Q + np.sqrt(self.c*np.log(self.i)*self.Sigmas), axis=0)
        return a

    def act_min(self):
        # Only works for U[0,1] distribution
        a = np.argmax(1-self.Q + np.sqrt(self.c*np.log(self.i)*self.Sigmas), axis=0)
        return a

    def backup(self, a, r, extra_info=None):
        self.n_obs[a, range(self.n_sim)] += 1
        self.i += 1
        if extra_info is None:
            # Case where bandit is at a leaf node
            self.V[a, range(self.n_sim)] += r
            self.W[a, range(self.n_sim)] += r*r
            self.Q[a, range(self.n_sim)] = self.V[a, range(self.n_sim)] / self.n_obs[a, range(self.n_sim)]
            Q_idx = self.Q[a, range(self.n_sim)]
            self.Sigmas[a, range(self.n_sim)] = self.W[a, range(self.n_sim)]/self.n_obs[a, range(self.n_sim)] - Q_idx*Q_idx  # Variance
        else:
            mu, sigma = extra_info
            assert(np.all(self.V == 0))
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma
        # Calculate own bandit distribution
        mu, sigma = self.calc_max_dist()
        return mu, sigma


class Bays_agent_Gauss_beta:
    def __init__(self, N_bandits, c=2, n_sim=1):
        self.Q = np.zeros((N_bandits, n_sim)) + np.float("inf")
        self.V = np.zeros((N_bandits, n_sim))  # Initialize with prior
        self.alpha = 1
        self.beta = 1
        self.Sigmas = np.ones((N_bandits, n_sim))  # These are sigma^2
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.N_bandits = N_bandits
        self.c = c
        self.i = 0
        self.n_sim = n_sim

    def combine(self, mu1, mu2, sigma1, sigma2):
        # Note sigma needs to be squared
        rho = 0
        sigma_m = sigma1+sigma2-2*rho*sigma1*sigma2
        alpha = (mu1-mu2)/sigma_m

        Phi_alpha = Phi(alpha)  # Store to avoid
        phi_alpha = phi(alpha)
        F1 = alpha*Phi_alpha + phi_alpha
        F2 = alpha*alpha*Phi_alpha*(1-Phi_alpha) + (1-2*Phi_alpha)*alpha*phi_alpha - phi_alpha*phi_alpha
        mu = mu2 + sigma_m*F1
        sigma = sigma2 + (sigma1-sigma2)*Phi_alpha + sigma_m * F2
        return mu, sigma

    def calc_max_dist(self):
        mu = self.Q[0][0]
        sigma = self.Sigmas[0][0]
        for i in range(1, self.N_bandits):
            mu, sigma = self.combine(mu, self.Q[i][0], sigma, self.Sigmas[i][0])
        return mu, sigma

    def act(self):
        a = np.argmax(self.Q + np.sqrt(self.c*np.log(self.i)*self.Sigmas), axis=0)
        return a

    def act_min(self):
        # Only works for U[0,1] distribution
        a = np.argmax(1-self.Q + np.sqrt(self.c*np.log(self.i)*self.Sigmas), axis=0)
        return a

    def backup(self, a, r, extra_info=None):
        self.n_obs[a, range(self.n_sim)] += 1
        self.i += 1
        if extra_info is None:
            # Case where bandit is at a leaf node
            self.V[a, range(self.n_sim)] += r
            alpha = self.alpha + self.V[a, range(self.n_sim)]
            beta = self.beta + self.n_obs[a, range(self.n_sim)] - self.V[a, range(self.n_sim)]
            self.Q[a, range(self.n_sim)] = alpha / (alpha + beta)
            self.Sigmas[a, range(self.n_sim)] = alpha*beta / ((alpha + beta)**2 * (alpha + beta + 1))
        else:
            mu, sigma = extra_info
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma
        # Calculate own bandit distribution
        mu, sigma = self.calc_max_dist()
        return mu, sigma


class Bays_agent_vector_UCT2:
    def __init__(self, N_bandits, support, c=2, n_sim=1):
        self.N_support = support.shape[0]
        self.theta = np.zeros((N_bandits, self.N_support)) + 1/self.N_support  # Initialize to uniform dist
        self.theta_count = np.zeros((N_bandits, self.N_support))# + 1/self.N_support  # To store each observation
        self.theta_count[:, 0] = 1
        self.theta_count[:, -1] = 1
        self.support = support
        self.support_squared = support*support
        self.Q = np.zeros((N_bandits, n_sim)) + np.float("inf")  # Average return for each child
        self.V = np.zeros((N_bandits, n_sim))  # Store sum of returns
        self.W = np.ones((N_bandits, n_sim))  # Store sum of squared returns
        self.Sigmas = np.ones((N_bandits, n_sim))  # These are sigma^2 of the child returns
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.N_bandits = N_bandits
        self.c = c
        self.i = 0
        self.n_sim = n_sim

    def combine(self, theta1, theta2):
        # Use cumsum for O(N) scaling. Alternative is outer product
        cdf1 = theta1.cumsum(axis=0)
        cdf2 = theta2.cumsum(axis=0)
        theta = cdf1*theta2 + cdf2*theta1 - theta1*theta2
        return theta

    def calc_max_dist(self):
        # calculate max distribution by combining all distirbutions iteratively
        theta = self.theta[0, :]
        for i in range(1, self.N_bandits):
            theta = self.combine(theta, self.theta[i, :])
        mean = (theta * self.support).sum()
        sigma = (theta * self.support_squared).sum() - mean*mean
        return mean, sigma, theta

    def act(self):
        a = np.argmax(self.Q + np.sqrt(self.c*np.log(self.i)*self.Sigmas), axis=0)
        return a

    def act_min(self):
        raise("Not implemented")

    @profile
    def backup(self, a, r, extra=None):
        self.n_obs[a, range(self.n_sim)] += 1
        self.i += 1
        if extra is None:
            # Add observation and normalize leaf
            lowest, highest = nearest_support(r, self.support)
            low_val = self.support[lowest]
            high_val = self.support[highest]
            lowest_p = (r - high_val) / (low_val - high_val)
            # Spread reward observation over support
            self.theta_count[a, lowest] += lowest_p
            self.theta_count[a, highest] += 1 - lowest_p

            # Try using beta prior
            beta = self.theta_count[a, 0]
            alpha = self.theta_count[a, -1]
            self.theta[a] = beta_to_theta(alpha, beta, self.support)

            # Case where bandit is at a leaf node
            mean_return = alpha / (alpha + beta)#(self.theta[a]*self.support).sum()
            self.Q[a, range(self.n_sim)] = mean_return
            self.Sigmas[a, range(self.n_sim)] = alpha*beta / ((alpha+beta)**2*(alpha+beta+1))#self.W[a, range(self.n_sim)]/self.n_obs[a, range(self.n_sim)] - Q_idx*Q_idx  # Variance
        else:
            # Case where bandit is not a leaf node
            mu, sigma, theta = extra
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma
            self.theta[a] = theta
        # Calculate own bandit distribution
        mu, sigma, theta = self.calc_max_dist()
        return mu, sigma, theta

def nearest_support(input, support):
    diffs = input - support
    diffs[diffs <= 0] = float("Inf")  # Only look for negative smallest values
    lowest = np.argmin(np.abs(diffs), axis=0)
    highest = lowest + 1
    return lowest, highest

def beta_to_theta(alpha, beta, support):
    theta = support**(alpha-1) * (1-support)**(beta-1)#scipy.stats.beta(alpha, beta)
    #theta = rv.pdf(support)
    theta = theta / theta.sum()
    return theta