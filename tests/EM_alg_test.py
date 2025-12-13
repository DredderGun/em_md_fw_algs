import numpy as np
import scipy.sparse as sp

# Import from your module:
# from algs import EM_algorithm
# from funcs import Topic_modeling_loss, Decorrelation_regularizer

from topic_modeling_alg_testing.algs.EM_alg import EM_algorithm
from topic_modeling_alg_testing.funcs.Topic_modelling import Topic_modeling_loss, Decorrelation_regularizer


# ------------------------------------------------------------
# Utility: create tiny synthetic dataset for deterministic tests
# ------------------------------------------------------------
def small_dataset():
    # D=3 documents, W=4 words, T=2 topics
    X = sp.csr_matrix([
        [3, 0, 1, 0],   # doc 1
        [0, 2, 0, 1],   # doc 2
        [1, 1, 0, 0],   # doc 3
    ], dtype=float)

    rng = np.random.default_rng(0)
    D, W = X.shape
    T = 2

    # simple initializations
    Phi0 = rng.dirichlet(np.ones(W), size=T).T
    Theta0 = rng.dirichlet(np.ones(T), size=D).T

    f_term = Topic_modeling_loss(X)
    reg = Decorrelation_regularizer(num_topics=T, tau=0.0)  # optional

    return X, Phi0.copy(), Theta0.copy(), f_term, reg


# ------------------------------------------------------------
# 1) Test shapes and normalization after one EM iteration
# ------------------------------------------------------------
def test_em_shapes_and_normalization():
    X, Phi, Theta, f_term, reg = small_dataset()
    Phi_new, Theta_new, F_hist, times = EM_algorithm(
        X, Phi, Theta, f_term, reg,
        maxitrs=3, verbose=False
    )

    W, T = Phi_new.shape
    T2, D = Theta_new.shape

    assert T == T2
    assert Phi_new.shape == (W, T)
    assert Theta_new.shape == (T, D)

    # Columns must sum to 1
    assert np.allclose(Phi_new.sum(axis=0), 1.0, atol=1e-8)
    assert np.allclose(Theta_new.sum(axis=0), 1.0, atol=1e-8)


# ------------------------------------------------------------
# 2) Test that objective does not increase
# ------------------------------------------------------------
def test_em_monotonic_decrease():
    X, Phi, Theta, f_term, reg = small_dataset()

    Phi_new, Theta_new, F_hist, _ = EM_algorithm(
        X, Phi, Theta, f_term, reg, 
        maxitrs=20, verbose=False
    )

    # EM should not increase objective (monotone)
    assert np.all(np.diff(F_hist) <= 1e-9)


# ------------------------------------------------------------
# 3) Test that EM actually changes Phi and Theta
# ------------------------------------------------------------
def test_em_parameter_updates():
    X, Phi0, Theta0, f_term, reg = small_dataset()

    Phi_new, Theta_new, F_hist, _ = EM_algorithm(
        X, Phi0.copy(), Theta0.copy(),
        f_term, reg,
        maxitrs=10, verbose=False
    )

    # Should not be equal (EM should move parameters)
    assert not np.allclose(Phi_new, Phi0)
    assert not np.allclose(Theta_new, Theta0)


# ------------------------------------------------------------
# 4) Test EM with regularizer (non-zero tau)
# ------------------------------------------------------------
def test_em_with_regularizer():
    X, Phi0, Theta0, f_term, _ = small_dataset()
    reg = Decorrelation_regularizer(num_topics=2, tau=5.0)

    Phi_new, Theta_new, F_hist, _ = EM_algorithm(
        X, Phi0.copy(), Theta0.copy(), 
        f_term, reg,
        maxitrs=10, verbose=False
    )

    # still correctly normalized
    assert np.allclose(Phi_new.sum(axis=0), 1.0, atol=1e-8)
    assert np.allclose(Theta_new.sum(axis=0), 1.0, atol=1e-8)


# ------------------------------------------------------------
# 5) Test EM on a trivial 1-doc, 1-word dataset
# ------------------------------------------------------------
def test_em_tiny():
    X = sp.csr_matrix([[10.0]])
    Phi = np.array([[1.0]])     # (1,1)
    Theta = np.array([[1.0]])            # (1,1)

    f_term = Topic_modeling_loss(X)
    reg = Decorrelation_regularizer(num_topics=1, tau=0.0)

    Phi_new, Theta_new, F_hist, _ = EM_algorithm(
        X, Phi.copy(), Theta.copy(),
        f_term, reg,
        maxitrs=5, verbose=False
    )

    assert Phi_new.shape == (1,1)
    assert Theta_new.shape == (1,1)
    assert np.allclose(Phi_new, 1.0)
    assert np.allclose(Theta_new, 1.0)
    assert len(F_hist) >= 1
