import numpy as np
import matplotlib.pyplot as plt


N_res = 6
N_tests = 10**6
theta1 = np.array([0, 0.3, 0, 0.5, 0.1, 0.1])
theta2 = np.array([0.2, 0.1, 0, 0.3, 0.2, 0.2])

# Outer product method
pred_dist = np.outer(theta1, theta2)
theta = np.zeros(theta1.shape)
for i in range(N_res):
    for j in range(N_res):
        theta[max(i, j)] += pred_dist[i, j]

# CDF method
cdf1 = theta1.cumsum()
cdf2 = theta2.cumsum()
theta_cdf = cdf1*theta2 + cdf2*theta1 - theta1*theta2

# Simulation
obs1 = np.random.choice(range(N_res), size=N_tests, p=theta1)
mean = (range(6)*theta1).sum()

obs2 = np.random.choice(range(N_res), size=N_tests, p=theta2)
obs = np.stack([obs1, obs2])
max_dist = np.max(obs, axis=0)

plt.hist(max_dist, density=True, bins=N_res, alpha=1./3, label="Sim")
plt.hist(np.array(range(N_res)), density=True, weights=theta, bins=N_res, alpha=1./3, label="Outer")
plt.hist(np.array(range(N_res)), density=True, weights=theta_cdf, bins=N_res, alpha=1./3, label="CDF")
plt.show()