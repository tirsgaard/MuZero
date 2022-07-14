import numpy as np
from scipy.stats import norm, beta
import math

def Phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def phi(x):
    # Propability mass distribution of the standard normal distribution
    return 0.3989422802*math.exp(-0.5*x*x)  # Number is 1/sqrt(2*pi)

def temperature_scale(P, temp, sd_noise, epsilon=10**-4):
    P = np.abs(np.random.normal(P, P * 0 + sd_noise))
    P = np.clip(P, epsilon, 1-epsilon)
    expon = 1/temp
    P_temp = P**expon
    P_temp = P_temp / P_temp.sum()
    return P_temp

def max_dist(mu1, mu2, sigma1, sigma2):
    # Normal approximation of max operation between two distributions
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
    def __init__(self, N_bandits, c=2, criteria="UCB1", temp=1, var_noise=0.0, n_sim=1, min_node=False):
        self.Q = np.zeros((N_bandits, n_sim)) #+ np.float("inf")
        self.V = np.zeros((N_bandits, n_sim))
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="int16")
        self.context = criteria
        self.min_node = min_node
        self.c = c
        self.temp = temp
        self.noise = var_noise  # Should be added outside agent, but this is easier
        self.i = 0
        self.c2 = 3
        self.n_sim = n_sim
        self.vl = 3
        s = temp
        self.epsilon = lambda x: 1/(x**s)
        self.explored = False
        if self.context == "UCB1":
            self.criterion = self.UCB1
        elif self.context == "PUCB":
            self.criterion = self.PUCT
        elif self.context == "MuZero":
            self.criterion = self.muZero
        elif self.context == "alphaZero":
            self.criterion = self.alphaZero
        elif self.context == 'epsilon':
            self.criterion = self.epsilon_greedy
        else:
            raise("Criterion not found")

    def muZero(self):
        c_1 = 1.25
        c_2 = 19652
        m = np.log((self.i+c_2+1)/c_2)
        temp = np.sqrt(self.i) / (1 + self.n_obs)
        a = np.argmax(self.Q + self.P * temp * (c_1 + m), axis=0)
        return a

    def epsilon_greedy(self):
        p = np.random.rand()
        if p<=self.epsilon(self.i+1):
            return np.array([np.random.choice(self.Q.shape[0])])
        else:
            return np.argmax(self.Q, axis=0)

    def alphaZero(self):
        c = 1.25
        m = np.sqrt(self.i)/(1+self.n_obs)
        a = np.argmax(self.Q + c * self.P * m, axis=0)
        return a

    def PUCT(self):
        t = self.i + 1  # This is 1 indexed
        log_t = np.log(t)
        c = np.sqrt((self.c2 / self.c) * log_t / self.n_obs)
        c[self.n_obs == 0] = 0
        m = (self.c / self.P) * np.sqrt(log_t / t) if t > 1 else self.c / self.P
        Q = self.Q.copy()
        Q[self.n_obs == 0] = np.float("inf")
        if self.min_node:
            a = np.argmin(Q - c + m, axis=0)
        else:
            a = np.argmax(Q + c - m, axis=0)
        return a

    def UCB1(self):
        if self.min_node:
            a = np.argmin(self.Q - np.sqrt(self.c * np.log(self.i) / self.n_obs), axis=0)
        else:
            a = np.argmax(self.Q + np.sqrt(self.c * np.log(self.i) / self.n_obs), axis=0)
        return a

    def act(self):
        return self.criterion()

    def get_greedy_action(self):
        if self.context == "PUCB":
            t = self.i + 1  # This is 1 indexed
            log_t = np.log(t)
            c = np.sqrt((self.c2 / self.c) * log_t / self.n_obs)
            c[self.n_obs == 0] = 0
            m = (self.c / self.P) * np.sqrt(log_t / t) if t > 1 else self.c / self.P
            a = np.argmax(self.Q - m)
        elif self.context == "MuZero":
            c_1 = 1.25
            c_2 = 19652
            m = np.log((self.i + c_2 + 1) / c_2)
            temp = np.sqrt(self.i) / (1 + self.i/self.n_obs.shape[0])
            a = np.argmax(self.Q + self.P * temp * (c_1 + m))

        elif self.context == "alphaZero":
            a = self.alphaZero()
        else:
            a = np.argmax(self.Q)
        return a

    def set_context(self, alphas, betas, is_leaf=False):
        if (self.context == "PUCB") or (self.context == "MuZero") or (self.context == "alphaZero"):
            # Compute chance of each child being the max
            x_range = np.linspace(0.00001, 0.99999, 10**4)  # Resolution of interation
            n_child = alphas.shape[0]
            pdfs = np.stack([beta.pdf(x_range, alphas[i], betas[i]) for i in range(n_child)])
            cdfs = np.stack([beta.cdf(x_range, alphas[i], betas[i]) for i in range(n_child)])

            self.P = np.empty((n_child, 1))
            for i in range(alphas.shape[0]):
                self.P[i] = np.mean(pdfs[i]*np.prod(cdfs[np.arange(n_child) != i], axis=0))  # Mean to reduce overflow
            self.P = self.P / self.P.sum()
            # Scale P with sqrt as it should yield the best asymtotic bound
            self.P = np.sqrt(self.P)
            self.P = self.P / self.P.sum()

    def get_dist(self):
        return None

    def initialize_dist(self, a, extra_info):
        return None

    def backup(self, a, r, extra_info=None, vl=False):
        self.explored = (not vl) or self.explored  # don't explore if virtual
        n_visit = self.vl if vl else 1

        self.V[a, range(self.n_sim)] += r*n_visit
        self.n_obs[a, range(self.n_sim)] += n_visit
        self.Q[a, range(self.n_sim)] = self.V[a, range(self.n_sim)] / self.n_obs[a, range(self.n_sim)]
        self.i += n_visit
        return extra_info

    def undo_vl(self, a, r, extra_info=None):
        self.V[a, range(self.n_sim)] -= r*self.vl
        self.n_obs[a, range(self.n_sim)] -= self.vl
        self.Q[a, range(self.n_sim)] = self.V[a, range(self.n_sim)] / self.n_obs[a, range(self.n_sim)]
        self.i -= self.vl
        return extra_info


class Bays_agent_Gauss_beta:
    def __init__(self, N_bandits, c=2, initialize=True, criteria="no_context", temp=1, var_noise=0.0, n_sim=1, min_node=False):
        self.V = np.zeros((N_bandits, n_sim))  # Initialize with prior
        self.context = criteria
        self.temp = temp
        self.noise = var_noise  # Should be added outside agent, but this is easier
        self.alpha = np.ones((N_bandits, n_sim))*2
        self.beta = np.ones((N_bandits, n_sim))*5
        self.Q = np.zeros((N_bandits, n_sim)) + self.get_mean(self.alpha, self.beta)
        self.Sigmas = np.zeros((N_bandits, n_sim)) + self.get_var(self.alpha, self.beta)  # These are sigma^2
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.N_bandits = N_bandits
        self.min_node = min_node
        self.initialize = initialize
        self.c = c
        self.c2 = 3  # For context
        self.i = 0
        self.vl = 3
        self.explored = False
        self.n_sim = n_sim
        if (self.context == "no_context") or (self.context == "UCT1_context") or (self.context == "UCT1_extracontext"):
            self.criterion = self.no_context
        elif self.context == "context":
            self.criterion = self.with_context
        elif self.context == "PUCT_context":
            self.criterion = self.PUCT
        elif self.context == "muzero_context":
            self.criterion = self.muZero
        elif self.context == "alphaGo":
            self.criterion = self.alphaGo
        elif self.context == "thompson":
            self.criterion = self.thompson
        else:
            raise("Criterion not found")

    def get_mean(self, alpha, beta):
        return alpha / (alpha + beta)

    def get_var(self, alpha, beta):
        return alpha*beta / ((alpha + beta)**2 * (alpha + beta + 1))

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

    def min_dist(self, mu1, mu2, sigma1, sigma2):
        # Note sigma needs to be squared
        rho = 0
        sigma_m = np.sqrt(sigma1+sigma2-2*rho*np.sqrt(sigma1*sigma2))
        alpha = -(mu1-mu2)/sigma_m

        Phi_alpha = Phi(alpha)  # Store to avoid
        phi_alpha = phi(alpha)
        F1 = alpha*Phi_alpha + phi_alpha
        F2 = alpha*alpha*Phi_alpha*(1-Phi_alpha) + (1-2*Phi_alpha)*alpha*phi_alpha - phi_alpha*phi_alpha
        mu = mu2 - sigma_m*F1
        sigma = sigma2 + (sigma1-sigma2)*Phi_alpha + sigma_m * sigma_m * F2
        return mu, sigma

    def calc_max_dist(self):
        mu = self.Q[0][0]
        sigma = self.Sigmas[0][0]
        for i in range(1, self.N_bandits):
            mu, sigma = self.max_dist(mu, self.Q[i][0], sigma, self.Sigmas[i][0])
        return mu, sigma

    def calc_min_dist(self):
        mu = self.Q[0][0]
        sigma = self.Sigmas[0][0]
        for i in range(1, self.N_bandits):
            mu, sigma = self.min_dist(mu, self.Q[i][0], sigma, self.Sigmas[i][0])
        return mu, sigma

    def set_context(self, alphas, betas, is_leaf=False):
        if (self.context == "UCT1_context") or (self.context == "thompson") or (self.context == "UCT1_extracontext"):
            alphabeta = alphas+betas
            self.Q[:, 0] = alphas/alphabeta
            self.Sigmas[:, 0] = alphas*betas/((alphabeta+1)*alphabeta*alphabeta)
            if is_leaf:
                self.alpha[:, 0] = alphas
                self.beta[:, 0] = betas
                #if self.context == "UCT1_extracontext":
                #    self.i += np.sum(self.alpha + self.beta)
                return
        if (self.context == "PUCT_context") or (self.context == "muzero_context"):
            # Compute chance of each child being the max
            x_range = np.linspace(0.00001, 0.99999, 10**4)  # Resolution of interation
            n_child = alphas.shape[0]
            pdfs = np.stack([beta.pdf(x_range, alphas[i], betas[i]) for i in range(n_child)])
            cdfs = np.stack([beta.cdf(x_range, alphas[i], betas[i]) for i in range(n_child)])

            self.P = np.empty((n_child, 1))
            for i in range(alphas.shape[0]):
                self.P[i] = np.mean(pdfs[i]*np.prod(cdfs[np.arange(n_child) != i], axis=0))  # Mean to reduce overflow
            self.P = self.P / self.P.sum()
            # Scale P with sqrt as it should yield the best asymtotic bound
            self.P = np.sqrt(self.P)
            self.P = self.P / self.P.sum()

    def no_context(self):
        if self.min_node:
            a = np.argmin(self.Q - np.sqrt(self.c * np.log(self.i) * self.Sigmas), axis=0)
        else:
            disc = np.log(self.i) if self.i>0 else float("inf")
            temp = self.Q + np.sqrt(self.c * disc * self.Sigmas)
            a = np.argmax(temp, axis=0)
        return a

    def with_context(self):
        t = self.i + 1  # This is 1 indexed
        log_t = np.log(t)
        c = np.sqrt((self.c2 / self.c) * log_t / self.n_obs)
        c[self.n_obs == 0] = 0
        m = (self.c / self.P) * np.sqrt(log_t / t) if t > 1 else self.c / self.P
        # Q[self.n_obs == 0] = 1
        if self.min_node:
            a = np.argmin(self.Q - c + m, axis=0)
        else:
            a = np.argmax(self.Q + c - m, axis=0)
        return a

    def PUCT(self):
        t = self.i + 1  # This is 1 indexed
        log_t = np.log(t)
        c = np.sqrt((self.c2 / self.c) * log_t /self.n_obs)
        c[self.n_obs == 0] = 0
        m = (self.c / self.P) * np.sqrt(log_t / t) if t > 1 else self.c / self.P
        Q = self.Q.copy()
        Q[self.n_obs == 0] = np.float("inf")
        if self.min_node:
            a = np.argmin(self.Q - c + m, axis=0)
        else:
            a = np.argmax(self.Q + c - m, axis=0)
        return a

    def alphaGo(self):
        c = 1.25
        m = np.sqrt(self.i) / (1 + self.n_obs)
        a = np.argmax(self.Q + c * self.P * m, axis=0)
        return a

    def thompson(self):
        a = np.argmax(np.random.normal(self.Q, np.sqrt(self.Sigmas)),  axis=0)
        return a

    def muZero(self):
        c_1 = 1.25
        c_2 = 19652
        m = np.log((self.i+c_2+1)/c_2)
        temp = np.sqrt(self.i) / (1 + self.n_obs)
        Q = self.Q.copy()
        Q[self.n_obs == 0] = np.float("inf")
        a = np.argmax(Q + self.P * temp * (c_1 + m), axis=0)
        return a

    def act(self):
        return self.criterion()

    def get_greedy_action(self):
        if (self.context == "no_context") or (self.context == "alphaGo") or (self.context == "thompson") or (self.context == "UCT1_context") or (self.context == "UCT1_extracontext"):
            a = np.argmax(self.Q)
        elif self.context == "context" or self.context == "PUCT_context":
            t = self.i + 1  # This is 1 indexed
            log_t = np.log(t)
            m = (self.c / self.P) * np.sqrt(log_t / t) if t > 1 else self.c / self.P
            a = np.argmax(self.Q - m)

        elif self.context == "muzero_context":
            c_1 = 1.25
            c_2 = 19652
            m = np.log((self.i + c_2 + 1) / c_2)
            temp = np.sqrt(self.i) / (1 + self.i / self.n_obs.shape[0])
            a = np.argmax(self.Q + self.P * temp * m)
        return a

    def get_dist(self):
        if self.min_node:
            mu, sigma = self.calc_min_dist()
        else:
            mu, sigma = self.calc_max_dist()
        return mu, sigma

    def initialize_dist(self, a, extra_info):
        if self.initialize and not self.context == "UCT1_context":
            mu, sigma = extra_info
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma

    def backup(self, a, r, extra_info=None, vl=False):
        self.explored = (not vl) or self.explored  # don't explore if virtual
        n_visits = self.vl if vl else 1
        self.n_obs[a, range(self.n_sim)] += n_visits
        self.i += n_visits
        if extra_info is None:
            # Case where bandit is at a leaf node
            self.V[a, range(self.n_sim)] += r*n_visits
            alpha = self.alpha[a, range(self.n_sim)] + self.V[a, range(self.n_sim)]
            beta = self.beta[a, range(self.n_sim)] + self.n_obs[a, range(self.n_sim)] - self.V[a, range(self.n_sim)]
            self.Q[a, range(self.n_sim)] = self.get_mean(alpha, beta)
            self.Sigmas[a, range(self.n_sim)] = self.get_var(alpha, beta)
        else:
            mu, sigma = extra_info
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma
        # Calculate own bandit distribution
        mu, sigma = self.get_dist()
        return mu, sigma

    def undo_vl(self, a, r, extra_info=None):
        self.n_obs[a, range(self.n_sim)] -= self.vl
        self.i -= self.vl
        if extra_info is None:
            # Case where bandit is at a leaf node
            self.V[a, range(self.n_sim)] -= self.vl*r
            alpha = self.alpha[a, range(self.n_sim)] + self.V[a, range(self.n_sim)]
            beta = self.beta[a, range(self.n_sim)] + self.n_obs[a, range(self.n_sim)] - self.V[a, range(self.n_sim)]
            self.Q[a, range(self.n_sim)] = self.get_mean(alpha, beta)
            self.Sigmas[a, range(self.n_sim)] = self.get_var(alpha, beta)
        else:
            mu, sigma = extra_info
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma
        # Calculate own bandit distribution
        mu, sigma = self.get_dist()
        return mu, sigma


class Bays_agent_vector_UCT2:
    def __init__(self, N_bandits, c=2, initialize=True, support=np.linspace(0, 1, 4000), criteria="max", n_sim=1, min_node=False):
        self.N_support = support.shape[0]
        self.theta = np.zeros((N_bandits, self.N_support))
        self.theta[:] = beta.pdf(support, 2, 5)[None]  #np.zeros((N_bandits, self.N_support)) + 1/self.N_support  # Initialize to uniform dist
        self.theta = self.theta / self.theta[0].sum()
        self.theta_count = np.zeros((N_bandits, self.N_support))  # + 1/self.N_support  # To store each observation
        self.theta_count[:, 0] = 5  # Beta
        self.theta_count[:, -1] = 2  # Alpha
        self.context = criteria
        self.support = support
        self.support_squared = support*support
        self.Q = np.zeros((N_bandits, n_sim)) + self.get_mean(2, 5)  # Average return for each child
        self.V = np.zeros((N_bandits, n_sim))  # Store sum of returns
        self.Sigmas = np.zeros((N_bandits, n_sim)) + self.get_var(2, 5)  # These are sigma^2 of the child returns
        self.n_obs = np.zeros((N_bandits, n_sim), dtype="uint16")
        self.initialize = initialize
        self.type = criteria
        self.min_node = min_node
        self.N_bandits = N_bandits
        self.c = c
        self.vl = 3
        self.explored = False
        self.temp = 1
        self.noise = 0.002551020408
        self.i = 0
        self.n_sim = n_sim
        if criteria == "UCT1":
            self.criteria = self.UCT1
        elif criteria == "thompson":
            self.criteria = self.thompson
        elif criteria == "thompson_bayes":
            self.criteria = self.thompson

    def max_dist(self, mu1, mu2):
        # Use cumsum for O(N) scaling. Alternative is outer product
        cdf1 = mu1.cumsum(axis=0)
        cdf2 = mu2.cumsum(axis=0)
        mu = cdf1*mu2 + cdf2*mu1 - mu1*mu2
        return mu

    def min_dist(self, mu1, mu2):
        # Use cumsum for O(N) scaling. Alternative is outer product
        cdf1 = 1 - mu1.cumsum(axis=0)
        cdf2 = 1 - mu2.cumsum(axis=0)
        mu = cdf1*mu2 + cdf2*mu1 - mu1*mu2
        return mu

    def get_mean(self, alpha, beta):
        return alpha / (alpha + beta)

    def get_var(self, alpha, beta):
        return alpha*beta / ((alpha + beta)**2 * (alpha + beta + 1))

    def calc_max_dist(self):
        # calculate max distribution by combining all distirbutions iteratively
        theta = self.theta[0, :]
        for i in range(1, self.N_bandits):
            theta = self.max_dist(theta, self.theta[i, :])
        mean = (theta * self.support).sum()
        sigma = (theta * self.support_squared).sum() - mean*mean
        return mean, sigma, theta

    def calc_min_dist(self):
        # calculate max distribution by combining all distirbutions iteratively
        theta = self.theta[0, :]
        for i in range(1, self.N_bandits):
            theta = self.min_dist(theta, self.theta[i, :])
        mean = (theta * self.support).sum()
        sigma = (theta * self.support_squared).sum() - mean*mean
        return mean, sigma, theta

    def initialize_dist(self, a, extra_info):
        if self.initialize:
            mu, sigma, theta = extra_info
            self.theta[a, :] = theta
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma

    def calc_thompson_dist(self):
        cdfs = np.cumsum(self.theta, axis=1)
        prod_cdfs = np.prod(cdfs, axis=0)
        prod_cdfs = prod_cdfs[None]/cdfs
        prod_cdfs[prod_cdfs!=prod_cdfs] = 0
        mixing_p = (self.theta*prod_cdfs).sum(axis=1, keepdims=True)
        mixing_p = mixing_p/mixing_p.sum()  # Should sum to one, but floating error
        theta = (mixing_p*self.theta).sum(axis=0)
        mean = (theta * self.support).sum()
        sigma = (theta * self.support_squared).sum() - mean * mean
        return mean, sigma, theta

    def get_dist(self):
        if self.type == "thompson_bayes":
            mu, sigma, theta = self.calc_thompson_dist()
        else:
            if self.min_node:
                mu, sigma, theta = self.calc_min_dist()
            else:
                mu, sigma, theta = self.calc_max_dist()
        return mu, sigma, theta

    def UCT1(self):
        disc = np.log(self.i) if self.i > 0 else 0
        if self.min_node:
            a = np.argmin(self.Q - np.sqrt(self.c * disc * self.Sigmas), axis=0)
        else:
            a = np.argmax(self.Q + np.sqrt(self.c * disc * self.Sigmas), axis=0)
        return a

    def thompson(self):
        # Sample from each distribution
        returns = np.empty((self.N_bandits,))
        for i in range(self.N_bandits):
            returns[i] = self.support[np.random.choice(self.theta[i].shape[0], p=self.theta[i])]
        a = np.argmax(returns, axis=0)[None]
        return a

    def act(self):
        return self.criteria()

    def set_context(self, alphas, betas, is_leaf=False):
        self.theta_count[:, 0] = betas
        self.theta_count[:, -1] = alphas
        if self.context == "UCT1":
            alphabeta = alphas + betas
            self.Q[:, 0] = alphas / alphabeta
            self.Sigmas[:, 0] = alphas * betas / ((alphabeta + 1) * alphabeta * alphabeta)
            if is_leaf:
                for a in range(alphas.shape[0]):
                    self.theta[a] = beta_to_theta(alphas[a], betas[a], self.support)

    def update_theta(self, a, r, vl=False, undo_vl=False):
        vl_multiplier = self.vl if vl else 1
        lowest, highest = nearest_support(r, self.support)
        low_val = self.support[lowest]
        high_val = self.support[highest]
        lowest_p = (r - high_val) / (low_val - high_val)
        # Spread reward observation over support
        if undo_vl:
            self.theta_count[a, lowest] -= vl_multiplier * lowest_p
            self.theta_count[a, highest] -= vl_multiplier * (1 - lowest_p)
        else:
            self.theta_count[a, lowest] += vl_multiplier*lowest_p
            self.theta_count[a, highest] += vl_multiplier*(1 - lowest_p)

    def get_greedy_action(self):
        a = np.argmax(self.Q)
        return a

    def backup(self, a, r, extra=None, vl=False):
        self.explored = (not vl) or self.explored  # don't explore if virtual
        n_visits = self.vl if vl else 1
        self.n_obs[a, range(self.n_sim)] += n_visits
        self.i += n_visits
        if extra is None:
            # Add observation and normalize leaf
            self.update_theta(a, r, vl=vl)
            # Try using beta prior
            beta = self.theta_count[a, 0]
            alpha = self.theta_count[a, -1]
            self.theta[a] = beta_to_theta(alpha, beta, self.support)
            # Case where bandit is at a leaf node
            mean_return = self.get_mean(alpha, beta)  #(self.theta[a]*self.support).sum()
            self.Q[a, range(self.n_sim)] = mean_return
            self.Sigmas[a, range(self.n_sim)] = alpha*beta / ((alpha+beta)**2*(alpha+beta+1))#self.W[a, range(self.n_sim)]/self.n_obs[a, range(self.n_sim)] - Q_idx*Q_idx  # Variance
        else:
            # Case where bandit is not a leaf node
            mu, sigma, theta = extra
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma
            self.theta[a] = theta
        # Calculate own bandit distribution
        if self.type == "thompson_bayes":
            mu, sigma, theta = self.calc_thompson_dist()
        else:
            if self.min_node:
                mu, sigma, theta = self.calc_min_dist()
            else:
                mu, sigma, theta = self.calc_max_dist()
        return mu, sigma, theta

    def undo_vl(self, a, r, extra_info=None):
        self.n_obs[a, range(self.n_sim)] -= self.vl
        self.i -= self.vl
        if extra_info is None:
            # Case where bandit is at a leaf node
            self.update_theta(a, r, vl=True, undo_vl=True)
            # Try using beta prior
            beta = self.theta_count[a, 0]
            alpha = self.theta_count[a, -1]
            self.theta[a] = beta_to_theta(alpha, beta, self.support)
            # Case where bandit is at a leaf node
            mean_return = self.get_mean(alpha, beta)  # (self.theta[a]*self.support).sum()
            self.Q[a, range(self.n_sim)] = mean_return
            self.Sigmas[a, range(self.n_sim)] = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        else:
            # Case where bandit is not a leaf node
            mu, sigma, theta = extra_info
            self.Q[a, range(self.n_sim)] = mu
            self.Sigmas[a, range(self.n_sim)] = sigma
            self.theta[a] = theta
        # Calculate own bandit distribution
        mu, sigma, theta = self.get_dist()
        return mu, sigma, theta

def nearest_support(input, support):
    diffs = input - support
    diffs[diffs <= 0] = float("Inf")  # Only look for negative smallest values
    lowest = np.argmin(np.abs(diffs), axis=0)
    highest = lowest + 1
    return lowest, highest

def beta_to_theta(alpha, beta, support):
    support = support.copy()
    support[0] += 10**-4
    support[-1] -= 10**-4
    theta = support**(alpha-1) * (1-support)**(beta-1)#scipy.stats.beta(alpha, beta)
    #theta = rv.pdf(support)
    theta = theta / theta.sum()
    return theta