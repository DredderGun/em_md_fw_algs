import numpy as np
import scipy.sparse as sp
import pytest

from topic_modeling_alg_testing.algs.Mirror_desc_alg import Mirror_descent_on_simplex


class QuadraticLoss:
    """
    f(Phi, Theta) = 0.5 * (||Phi - Phi_star||^2 + ||Theta - Theta_star||^2)
    """
    def __init__(self, Phi_star, Theta_star):
        self.Phi_star = Phi_star
        self.Theta_star = Theta_star

    def __call__(self, Phi, Theta):
        return 0.5 * (np.sum((Phi - self.Phi_star)**2) +
                      np.sum((Theta - self.Theta_star)**2))

    def gradient_by_phi(self, Phi, Theta):
        return Phi - self.Phi_star

    def gradient_by_theta(self, Phi, Theta):
        return Theta - self.Theta_star


class ZeroLoss:
    def __call__(self, Phi, Theta):
        return 0.0

    def gradient_by_phi(self, Phi, Theta):
        return np.zeros_like(Phi)

    def gradient_by_theta(self, Phi, Theta):
        return np.zeros_like(Theta)
    

class ZeroRegularizer:
    def gradient(self, Phi):
        return np.zeros_like(Phi)


@pytest.fixture
def small_simplex_problem():
    rng = np.random.default_rng(0)

    num_terms = 5
    num_topics = 3
    num_docs = 4

    Phi0 = rng.dirichlet(np.ones(num_terms), size=num_topics).T
    Theta0 = rng.dirichlet(np.ones(num_topics), size=num_docs).T

    Phi_star = rng.dirichlet(np.ones(num_terms), size=num_topics).T
    Theta_star = rng.dirichlet(np.ones(num_topics), size=num_docs).T

    X = sp.csr_matrix((num_terms, num_docs))

    loss = QuadraticLoss(Phi_star, Theta_star)
    reg = ZeroRegularizer()

    return X, Phi0, Theta0, Phi_star, Theta_star, loss, reg


def test_shapes_preserved(small_simplex_problem):
    X, Phi0, Theta0, _, _, loss, reg = small_simplex_problem

    Phi, Theta, _, _ = Mirror_descent_on_simplex(
        X, Phi0, Theta0, loss, reg, maxitrs=5, verbose=False
    )

    assert Phi.shape == Phi0.shape
    assert Theta.shape == Theta0.shape


def test_simplex_invariance(small_simplex_problem):
    X, Phi0, Theta0, _, _, loss, reg = small_simplex_problem

    Phi, Theta, _, _ = Mirror_descent_on_simplex(
        X, Phi0, Theta0, loss, reg, maxitrs=10, verbose=False
    )

    # Columns sum to 1
    assert np.allclose(Phi.sum(axis=0), 1.0, atol=1e-9)
    assert np.allclose(Theta.sum(axis=0), 1.0, atol=1e-9)

    # Non-negativity
    assert np.all(Phi >= 0)
    assert np.all(Theta >= 0)


def test_no_nans_or_infs(small_simplex_problem):
    X, Phi0, Theta0, _, _, loss, reg = small_simplex_problem

    Phi, Theta, F_hist, _ = Mirror_descent_on_simplex(
        X, Phi0, Theta0, loss, reg, maxitrs=20, verbose=False
    )

    assert np.isfinite(Phi).all()
    assert np.isfinite(Theta).all()
    assert np.isfinite(F_hist).all()


def test_zero_gradient_identity():
    rng = np.random.default_rng(1)
    Phi0 = rng.dirichlet(np.ones(5), size=3).T
    Theta0 = rng.dirichlet(np.ones(3), size=4).T

    X = sp.csr_matrix((5, 4))
    loss = ZeroLoss()
    reg = ZeroRegularizer()

    Phi, Theta, _, _ = Mirror_descent_on_simplex(
        X, Phi0, Theta0, loss, reg, maxitrs=5, verbose=False
    )

    assert np.allclose(Phi, Phi0)
    assert np.allclose(Theta, Theta0)


def test_objective_decreases(small_simplex_problem):
    X, Phi0, Theta0, _, _, loss, reg = small_simplex_problem

    _, _, F_hist, _ = Mirror_descent_on_simplex(
        X, Phi0, Theta0, loss, reg, maxitrs=20, verbose=False
    )

    # Allow small numerical noise, but trend must be decreasing
    assert F_hist[-1] <= F_hist[0] + 1e-8


def test_determinism(small_simplex_problem):
    X, Phi0, Theta0, _, _, loss, reg = small_simplex_problem

    Phi1, Theta1, _, _ = Mirror_descent_on_simplex(
        X, Phi0, Theta0, loss, reg, maxitrs=10, verbose=False
    )

    Phi2, Theta2, _, _ = Mirror_descent_on_simplex(
        X, Phi0, Theta0, loss, reg, maxitrs=10, verbose=False
    )

    assert np.allclose(Phi1, Phi2)
    assert np.allclose(Theta1, Theta2)


def test_large_values_stability(small_simplex_problem):
    X, Phi0, Theta0, _, _, loss, reg = small_simplex_problem

    Phi0 *= 1e2
    Phi0 /= Phi0.sum(axis=0, keepdims=True)

    Phi, Theta, _, _ = Mirror_descent_on_simplex(
        X, Phi0, Theta0, loss, reg, maxitrs=30, verbose=False
    )

    assert np.isfinite(Phi).all()
    assert np.allclose(Phi.sum(axis=0), 1.0, atol=1e-9)

