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

    problem_size = max(q.shape)
    solution_variable = cp.Variable(n)

    objective = cp.Minimize(q.T @ solution_variable + 0.5 * cp.quad_form(solution_variable, Q))
    constraints = [solution_variable <= b]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return solution_variable.value
