import numpy as np
import time
from scipy.sparse import coo_matrix
import pickle
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

from .common import SmoothFunction

class Topic_modeling_loss(SmoothFunction):
    def __init__(self, X: sp.csr_matrix):
        self.X = X.tocsr()      # (D, W)

    def __call__(self, Phi: np.ndarray, Theta: np.ndarray) -> float:
        # Phi:   (W, T)
        # Theta: (T, D)
        num_docs, num_terms = self.X.shape
        eps = 1e-12
        loss = 0.0

        for d in range(num_docs):
            start, end = self.X.indptr[d], self.X.indptr[d+1]
            cols = self.X.indices[start:end]    # word indices
            vals = self.X.data[start:end]       # counts

            theta_d = Theta[:, d]               # (T,)
            phi_cols = Phi[cols, :]             # (K, T)

            pw = phi_cols @ theta_d             # (K,)
            pw = np.maximum(pw, eps)

            loss -= np.sum(vals * np.log(pw))

        return loss

    def gradient_by_theta(self, Phi, Theta):
        num_docs, num_terms = self.X.shape
        T = Theta.shape[0]
        grad_theta = np.zeros_like(Theta)
        eps = 1e-12

        for d in range(num_docs):
            start, end = self.X.indptr[d], self.X.indptr[d+1]
            cols = self.X.indices[start:end]
            vals = self.X.data[start:end]

            theta_d = Theta[:, d]          # (T,)
            phi_cols = Phi[cols, :]        # (K, T)

            pw = phi_cols @ theta_d        # (K,)
            pw = np.maximum(pw, eps)

            # (K, T) * (K,1) → (K,T), then sum over words
            contrib = (phi_cols / pw[:, None]) * vals[:, None]   # (K,T)

            grad_theta[:, d] -= contrib.sum(axis=0)              # (T,)

        return grad_theta

    def gradient_by_phi(self, Phi, Theta):
        num_docs, num_terms = self.X.shape
        grad_phi = np.zeros_like(Phi)
        eps = 1e-12

        for d in range(num_docs):
            start, end = self.X.indptr[d], self.X.indptr[d+1]
            cols = self.X.indices[start:end]
            vals = self.X.data[start:end]

            theta_d = Theta[:, d]             # (T,)
            phi_cols = Phi[cols, :]           # (K, T)

            # (T,) @ (T,K) = (K,)   → need phi_cols.T
            pw = phi_cols @ theta_d           # (K,)
            pw = np.maximum(pw, eps)

            # gradient wrt each φ(w,t)
            # shape: (K,T) = (vals/pw)[:,None] * theta_d[None,:]
            contrib = (vals / pw)[:, None] * theta_d[None, :]   # (K, T)

            grad_phi[cols, :] -= contrib       # scatter-add into rows of Φ

        return grad_phi
    

class Decorrelation_regularizer(SmoothFunction):
    def __init__(self, num_topics: int, tau: float = 1.0):
        self.num_topics = num_topics
        self.tau = tau

    def __call__(self, Phi: np.ndarray) -> float:
        # R(Phi) = - τ / (T(T-1)) * sum_{t≠s} sum_w phi[w,t] phi[w,s]

        G = Phi @ Phi.T
        offdiag_sum = G.sum() - np.trace(G)  # sum_{t≠s} G_ts

        R_phi = -self.tau / (self.num_topics * (self.num_topics - 1)) * offdiag_sum

        return R_phi
    
    def gradient(self, Phi: np.ndarray):
        # derivative w.r.t phi_uw: dR/dphi_uw = -2*alpha*(S_w - phi_uw), where S_w = sum_t phi_tw
        try:
            alpha = self.tau / (self.num_topics * (self.num_topics - 1.0))
            S_w = Phi.sum(axis=1)  # shape (W,)
            grad_phi = -2 * alpha * (S_w[:, None] - Phi)  # (W,T)
        except Exception:
            # If reg_term has no .tau/.num_topics, assume 0 regularization (no gradient)
            grad_phi = np.zeros_like(Phi, dtype=float)
        return grad_phi


    def func_grad(self, x, flag):
        if flag == 0:
            return self.__call__(x)
        elif flag == 1:
            return self.gradient(x)
        elif flag == 2:
            return self.__call__(x), self.gradient(x)
        else:
            assert 0, "Decorrelation_regularizer: func_grad(x, flag) invalid flag"
