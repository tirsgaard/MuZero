import torch
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import norm

def trans_conv_prepare(support, inv_func):
    trans_supp = inv_func(torch.from_numpy(support)).numpy().astype(np.float64)
    #plt.plot(support, np.log(trans_supp))
    sum_supp = trans_supp[:, None]  + trans_supp[None]
    sum_supp = np.reshape(sum_supp, (-1,))

    # Expects input to be of shape (B, K) and support in increasing order of shape (N)
    # Returns index of low and high value of index
    diffs = sum_supp[:, None] - trans_supp[None]
    diffs[diffs <= 0.0] = float("Inf")  # Only look for negative smallest values
    lowest_idx = np.argmin(diffs, axis=1).astype(np.int64)
    highest_idx = np.clip(lowest_idx + 1, None, support.shape[0]-1).astype(np.int64)  # Map values outside support back into support


    low_val = trans_supp[lowest_idx].clip(trans_supp[0], trans_supp[-2])
    high_val = trans_supp[highest_idx].clip(trans_supp[1], trans_supp[-1])
    lowest_p = (sum_supp.clip(trans_supp[0], trans_supp[-1]) - high_val) / (low_val - high_val)

    sup_sq = lowest_p.shape[0]
    large_mat = lil_matrix((support.shape[0], sup_sq))
    large_mat[lowest_idx, range(sup_sq)] = lowest_p
    large_mat[highest_idx, range(sup_sq)] = 1 - lowest_p
    large_mat = csr_matrix(large_mat)
    return large_mat


def sum_dist(dist1, dist2, add_mat):
    outer_prod = dist1[:, None] * dist2[None]
    outer_prod = np.reshape(outer_prod, (-1,))
    dist = add_mat.dot(outer_prod)
    return dist


def normal_support(mu, sigma, support_trans):
    # Function for converting normal distribution to support distribution
    support_len = np.zeros(support_trans.shape + (1, )*len(mu.shape))
    for i in range(support_len.shape[0]-1):
        support_len[i] = (support_trans[i] + support_trans[i+1])/2
    support_len[-1] = np.float("inf")

    dist2 = norm.cdf(support_len, loc=mu, scale=np.sqrt(sigma))
    dist = np.zeros(dist2.shape)
    dist[0] = dist2[0]
    for i in range(1, support_len.shape[0]-1):
        dist[i] = dist2[i] - dist2[i-1]
    dist[-1] = 1 - dist2[-2]
    dist = np.moveaxis(dist, 0, -1)
    return torch.from_numpy(dist)