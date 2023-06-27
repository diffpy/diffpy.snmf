import numpy as np
import cvxpy as cp


def mkr_box(stretched_component_gram_matrix, linear_coefficient, lower_bound, upper_bound):
    """ Finds the weightings of stretched component signals

    Solves min J(x) = (linear_coefficient)' * x + (1/2) * x' * (quadratic coefficient) * x where lower_bound <= x <=
    upper_bound and stretched_component_gram_matrix is symmetric positive definite

    Parameters
    ----------
    stretched_component_gram_matrix: 2d array like
      The Gram matrix constructed from the stretched component matrix. It is a square positive definite matrix. It has
      dimensions C x C where C is the number of component signals.

    linear_coefficient: 1d array like
      The vector containing

    lower_bound: 1d array like
      The lower limit on the value of the output element wise. Has length C

    upper_bound: 1d array like
      The upper limit on the values of the output element wise. Has length C.

    Returns
    -------
    1d array like
      The vector containing the weightings of the component signal at a certain moment. Has length C.

    Raises
    ------
    ValueError
        If stretched_component_gram_matrix is not a Symmetric Positive Definite Matrix.

    """
    stretched_component_gram_matrix = np.asarray(stretched_component_gram_matrix)
    linear_coefficient = np.asarray(linear_coefficient)
    upper_bound = np.asarray(upper_bound)
    lower_bound = np.asarray(lower_bound)

    problem_size = max(linear_coefficient.shape)
    solution_variable = cp.Variable(problem_size)

    objective = cp.Minimize(
        linear_coefficient.T @ solution_variable + 0.5 * cp.quad_form(solution_variable, stretched_component_gram_matrix))
    constraints = [lower_bound <= solution_variable, solution_variable <= upper_bound]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return solution_variable.value
