import numpy as np
import cvxpy as cp


def mkr_box(quadratic_coefficient, linear_coefficient, lower_bound, upper_bound):
    """
    Solves min J(x) = (linear_coefficient)' * x + (1/2) * x' * (quadratic coefficient) * x where lower_bound <= x <=
    upper_bound and quadratic_coefficient is symmetric positive definite

    Parameters
    ----------
    quadratic_coefficient
    linear_coefficient
    lower_bound
    upper_bound

    Returns
    -------

    """
    quadratic_coefficient = np.asarray(quadratic_coefficient)
    linear_coefficient = np.asarray(linear_coefficient)
    upper_bound = np.asarray(upper_bound)
    lower_bound = np.asarray(lower_bound)

    problem_size = max(linear_coefficient.shape)
    solution_variable = cp.Variable(problem_size)

    objective = cp.Minimize(
        linear_coefficient.T @ solution_variable + 0.5 * cp.quad_form(solution_variable, quadratic_coefficient))
    constraints = [lower_bound <= solution_variable, solution_variable <= upper_bound]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return solution_variable.value
