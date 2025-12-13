import numpy as np
import scipy.sparse as sp

from topic_modeling_alg_testing.funcs.common import SmoothFunction

def EM_algorithm(X: sp.csr_matrix, Phi: np.ndarray, Theta: np.ndarray,
                 f_term: SmoothFunction, reg_term: SmoothFunction,
                 maxitrs=1000, verbskip=10, verbose=True, name="EM") -> tuple[np.ndarray, np.ndarray]:
    if verbose:
        print(f"\n{name} Algorithm")
        print("     k        F(x)        time")
    eps = 1e-12

    X = X.tocsr()
    num_docs, num_terms = X.shape
    _, num_topics = Phi.shape
    assert Phi.shape == (num_terms, num_topics)
    assert Theta.shape == (num_topics, num_docs)
    assert np.allclose(Phi.sum(axis=0), 1.0, atol=1e-8)
    assert np.allclose(Theta.sum(axis=0), 1.0, atol=1e-8)

    F_hist = []
    time_hist = []

    import time as _time
    t0 = _time.time()

    data_loss = f_term(Phi, Theta)
    try:
        reg_val = reg_term(Phi)
    except Exception:
        reg_val = 0.0
    # F = data_loss + reg_val
    F = data_loss
    F_hist.append(F)
    time_hist.append(0.0)  

    for k in range(maxitrs):
        iter_start = _time.time()

        n_wt = np.zeros_like(Phi, dtype=float)  # (W, T)
        n_td = np.zeros((num_topics, num_docs), dtype=float) # (T, D)

        for d in range(num_docs):
            start = X.indptr[d]
            end = X.indptr[d+1]
            if start == end:
                continue  # empty document
            cols = X.indices[start:end] # word indices in doc d
            vals = X.data[start:end].astype(float) # counts for those words

            Phi_sub_rows = Phi[cols, :] # (n, T) words in doc d
            Theta_sub_cols = Theta[:, d] # (T,) topic dist for doc d

            p_tw_for_d = Phi_sub_rows * Theta_sub_cols[np.newaxis, :] # here we started collect p_tdw for doc d
            p_tw_for_d /= np.maximum(p_tw_for_d.sum(axis=1), eps)[:, np.newaxis] 

            weighted = p_tw_for_d * vals[:, np.newaxis]
            
            for j, w in enumerate(cols):
                n_wt[w, :] += weighted[j, :]

            n_td[:, d] = weighted.sum(axis=0)

        grad_phi = reg_term.gradient(Phi)
        Phi_update = n_wt + Phi * grad_phi        # (W, T)
        Phi_update = np.maximum(Phi_update, eps)

        col_sums = Phi_update.sum(axis=0)         # (T,)
        for t in range(num_topics):
            # вероятность равна 0/0, поэтому, чтобы не было ошибки,
            # инициализируем любым распределением, все равно каким
            if col_sums[t] <= eps:
                Phi[:, t] = (np.ones(num_terms) / float(num_terms)) + eps
            else:
                Phi[:, t] = Phi_update[:, t] / col_sums[t]

        # ATTENTION!!! Градиент по Theta не реализован вообще! Если в регуляризаторе есть градиент по Theta, он не учитывается
        Theta_update = n_td.copy()
        col_sums = Theta_update.sum(axis=0)
        for d in range(num_docs):
            if col_sums[d] <= eps:
                Theta[:, d] = (np.ones(num_topics) / float(num_topics)) + eps
            else:
                Theta[:, d] = Theta_update[:, d] / col_sums[d]

        data_loss = f_term(Phi, Theta)
        try:
            reg_val = reg_term(Phi)
        except Exception:
            reg_val = 0.0
        # F = data_loss + reg_val
        F = data_loss

        F_hist.append(F)
        time_hist.append(_time.time() - t0)

        if verbose and (k % verbskip == 0 or k == maxitrs - 1):
            print(f"{k:6d}  {F:12.6e}  {time_hist[-1]:6.1f}")

        if k > 0:
            rel_change = abs(F_hist[-1] - F_hist[-2])
            if rel_change < 1e-9:
                if verbose:
                    print(f"converged (rel change {rel_change:.2e}) at iter {k}")
                break   
    
    return Phi, Theta, np.array(F_hist), np.array(time_hist)