import numpy as np


def rooth(linear_coefficient, constant_term):
    """
    Returns the largest real root of x^3+(linear_coefficient) * x + constant_term. If there are no real roots
    return 0.

    Parameters
    ----------
    linear_coefficient: nd array like of floats
        The matrix coefficient of the linear term
    constant_term: 0d array like, 1d array like of floats or scalar
        The constant scalar term of the problem

    Returns
    -------
    ndarray of floats
        The largest real root of x^3+(linear_coefficient) * x + constant_term if roots are real, else
        return 0 array


    """
    linear_coefficient = np.asarray(linear_coefficient)
    constant_term = np.asarray(constant_term)
    solution = np.empty_like(linear_coefficient, dtype=np.float64)

    for index, value in np.ndenumerate(linear_coefficient):
        inputs = [1, 0, value, constant_term]
        roots = np.roots(inputs)
        if ((constant_term / 2) ** 2 + (value / 3) ** 3) < 0:  # Discriminant of depressed cubic equation
            solution[index] = max(np.real(roots))
        else:
            solution[index] = 0
    return solution
