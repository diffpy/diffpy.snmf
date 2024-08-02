import cvxpy
import numpy as np


def get_weights(stretched_component_gram_matrix, linear_coefficient, lower_bound, upper_bound):
    """Finds the weights of stretched component signals under a two-sided constraint

    Solves min J(y) = (linear_coefficient)' * y + (1/2) * y' * (quadratic coefficient) * y where
    lower_bound <= y <= upper_bound and stretched_component_gram_matrix is symmetric positive definite.
    Finds the weightings of stretched component signals under a two-sided constraint.

    Parameters
    ----------
    stretched_component_gram_matrix: 2d array like
      The Gram matrix constructed from the stretched component matrix. It is a square positive definite matrix.
      It has dimensions C x C where C is the number of component signals. Must be symmetric positive definite.

    linear_coefficient: 1d array like
      The vector containing the product of the stretched component matrix and the transpose of the observed
      data matrix. Has length C.

    lower_bound: 1d array like
      The lower bound on the values of the output weights. Has the same dimensions of the function output. Each
      element in 'lower_bound' determines the minimum value the corresponding element in the function output may
      take.

    upper_bound: 1d array like
      The upper bound on the values of the output weights. Has the same dimensions of the function output. Each
      element in 'upper_bound' determines the maximum value the corresponding element in the function output may
      take.

    Returns
    -------
    1d array like
      The vector containing the weightings of the components needed to reconstruct a given input signal from the
      input set. Has length C

    """
    stretched_component_gram_matrix = np.asarray(stretched_component_gram_matrix)
    linear_coefficient = np.asarray(linear_coefficient)
    upper_bound = np.asarray(upper_bound)
    lower_bound = np.asarray(lower_bound)

    problem_size = max(linear_coefficient.shape)
    solution_variable = cvxpy.Variable(problem_size)

    objective = cvxpy.Minimize(
        linear_coefficient.T @ solution_variable
        + 0.5 * cvxpy.quad_form(solution_variable, stretched_component_gram_matrix)
    )
    constraints = [lower_bound <= solution_variable, solution_variable <= upper_bound]

    cvxpy.Problem(objective, constraints).solve()

    return solution_variable.value
