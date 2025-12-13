import numpy as np
import scipy.sparse as sp

from topic_modeling_alg_testing.funcs.common import SmoothFunction

def Mirror_descent_on_simplex(X: sp.csr_matrix, Phi0: np.ndarray, Theta0: np.ndarray,
                 topic_model_loss: SmoothFunction, reg_term: SmoothFunction,
                 maxitrs=1000, verbskip=10, epsilon=1e-14, verbose=True, name="") -> tuple[np.ndarray, np.ndarray]:
    if verbose:
        print(r"\nMirror descent on simplex, name={}".format(name))
        print("     k        F(x)        time")

    X = X.tocsr()
    
    F_hist = []
    time_hist = []

    import time as _time
    t0 = _time.time()

    Theta = np.copy(Theta0)
    Phi = np.copy(Phi0)
    data_loss = topic_model_loss(Phi, Theta)
    F_hist.append(data_loss)

    alpha0 = 1e-2
    for k in range(maxitrs):
        assert np.allclose(Phi.sum(axis=0), 1.0, atol=1e-9), r'Phi columns do not sum to 1, mean sum is {}'.format(np.mean(Phi.sum(axis=0)))
        assert np.allclose(Theta.sum(axis=0), 1.0, atol=1e-9)

        iter_start = _time.time()
        # alpha_k = 1e-2 / (np.sqrt(k+1))

        # ---------------------- Θ STEP ----------------------
        f_grad_theta = topic_model_loss.gradient_by_theta(Phi, Theta)
        # alpha_k = alpha0 / (np.linalg.norm(f_grad_theta, ord=np.inf, axis=0).max() + 1e-12)
        alpha_k = alpha0 / (np.max(np.abs(f_grad_theta)) + 1e-12)

        z = -alpha_k * f_grad_theta
        z -= np.max(z, axis=0, keepdims=True) # stability shift

        num = np.exp(z) * Theta  # numerator, elementwise
        den = np.sum(num, axis=0, keepdims=True)
        den[den < 1e-12] = 1e-12  # avoid division by zero
        Theta = num / den

        # ---------------------- Φ STEP ----------------------
        f_grad_phi = topic_model_loss.gradient_by_phi(Phi, Theta)
        reg_grad_phi = reg_term.gradient(Phi)
        total_grad_phi = f_grad_phi + reg_grad_phi
        # alpha_k = alpha0 / (np.linalg.norm(total_grad_phi, ord=np.inf) + 1e-12)
        alpha_k = alpha0 / (np.max(np.abs(total_grad_phi)) + 1e-12)
        
        z = -alpha_k * total_grad_phi
        z -= np.max(z, axis=0, keepdims=True)  # stability shift
        num = np.exp(z) * Phi
        den = np.sum(num, axis=0, keepdims=True)
        den[den < 1e-12] = 1e-12  # avoid division by zero
        Phi = num / den

        # ---------------------- Logging ----------------------
        fx = topic_model_loss(Phi, Theta)
        F_hist.append(fx)
        time_hist.append(_time.time() - t0)

        if verbose and (k % verbskip == 0 or k == 1):
            print(f"{k:6d}  {F_hist[k]:12.6e}  {time_hist[-1]:6.1f}")
        
        if k > 1 and abs(F_hist[k] - F_hist[k-1]) < epsilon:
            break

    return Phi, Theta, np.array(F_hist), np.array(time_hist)
