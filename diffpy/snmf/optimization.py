import numpy as np
import cvxpy as cp


def mkr_box(Q, q, b):
    """
    Solves min J(x) = q'x + (1/2) * x'Qx where x <= b and Q is symmetric positive definite

    Parameters
    ----------
    Q
    q
    b

    Returns
    -------

    """
    Q = np.asarray(Q)
    q = np.asarray(q)
    b = np.asarray(b)

    n = Q.shape[0]
    x = cp.Variable(n)

    objective = cp.Minimize(q.T @ x + 0.5 * cp.quad_form(x, Q))
    constraints = [x <= b]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return x.value
