import numpy as np
import scipy.sparse
from pathlib import Path
from diffpy.utils.parsers.loaddata import loadData
import matplotlib.pyplot as plt
from bg_mpl_stylesheet.bg_mpl_stylesheet import bg_mpl_style

plt.style.use(bg_mpl_style)


def initialize_variables(data_input, component_amount, data_type, sparsity=1, smoothness=1e18):
    """Determines the variables and initial values used in the SNMF algorithm.

    Parameters
    ----------
    data_input: 2d array like
      The observed or simulated PDF or XRD data provided by the user. Has dimensions R x N where R is the signal length
      and N is the number of PDF/XRD signals.

    component_amount: int
      The number of component signals the user would like to decompose 'data_input' into.

    data_type: str
      The type of data the user has passed into the program. Can assume the value of 'PDF' or 'XRD.'

    sparsity: float, optional
      The regularization parameter that behaves as the coefficient of a "sparseness" regularization term that enhances
      the ability to decompose signals in the case of sparse data e.g. X-ray Diffraction data. A non-zero value
      indicates sparsity in the data; greater magnitudes indicate greater amounts of sparsity.

    smoothness: float, optional
      The regularization parameter that behaves as the coefficient of a "smoothness" term that ensures that component
      signal weightings change smoothly with time. Assumes a default value of 1e18.

    Returns
    -------
    dictionary
      The collection of the names and values of the constants used in the algorithm. Contains the number of observed PDF
      /XRD patterns, the length of each pattern, the type of the data, the number of components the user would like to
      decompose the data into, an initial guess for the component matrix, and initial guess for the weight factor matrix
      ,an initial guess for the stretching factor matrix, a parameter controlling smoothness of the solution, a
      parameter controlling sparseness of the solution, the matrix representing the smoothness term, and a matrix used
      to construct a hessian matrix.

    """
    signal_length = data_input.shape[0]
    moment_amount = data_input.shape[1]

    component_matrix_guess = np.random.rand(signal_length, component_amount)
    weight_matrix_guess = np.random.rand(component_amount, moment_amount)
    stretching_matrix_guess = np.ones(component_amount, moment_amount) + np.random.randn(component_amount,
                                                                                         moment_amount) * 1e-3

    diagonals = [np.ones(moment_amount - 2), -2 * np.ones(moment_amount - 2), np.ones(moment_amount - 2)]
    smoothness_term = .25 * scipy.sparse.diags(diagonals, [0, 1, 2], shape=(moment_amount - 2, moment_amount))

    hessian_helper_matrix = scipy.sparse.block_diag([smoothness_term.T @ smoothness_term] * component_amount)
    sequence = np.arange(moment_amount * component_amount).reshape(component_amount, moment_amount).T.flatten()
    hessian_helper_matrix = hessian_helper_matrix[sequence, :][:, sequence]

    return {
        "signal_length": signal_length,
        "moment_amount": moment_amount,
        "component_matrix_guess": component_matrix_guess,
        "weight_matrix_guess": weight_matrix_guess,
        "stretching_matrix_guess": stretching_matrix_guess,
        "component_amount": component_amount,
        "data_type": data_type,
        "smoothness": smoothness,
        "sparsity": sparsity,
        "smoothness_term": smoothness_term,
        "hessian_helper_matrix": hessian_helper_matrix
    }


def load_input_signals(file_path=None):
    """Processes a directory of a series of PDF/XRD patterns into a usable format.

    Constructs a 2d array out of a directory of PDF/XRD patterns containing each files dependent variable column in a
    new column. Constructs a 1d array containing the grid values.

    Parameters
    ----------
    file_path: str or Path object, optional
      The path to the directory containing the input XRD/PDF data. If no path is specified, defaults to the current
      working directory. Accepts a string or a pathlib.Path object. Input data not on the same grid as the first file
      read will be ignored.

    Returns
    -------
    tuple
      The tuple whose first element is an R x M 2d array made of PDF/XRD patterns as each column; R is the length of the
      signal and M is the number of patterns. The tuple contains a 1d array containing the values of the grid points as
      its second element; Has length R.

    """

    if file_path is None:
        directory_path = Path.cwd()
    else:
        directory_path = Path(file_path)

    values_list = []
    grid_list = []
    current_grid = []
    for item in directory_path.iterdir():
        if item.is_file():
            data = loadData(item.resolve())
            if current_grid and current_grid != data[:, 0]:
                print(f"{item.name} was ignored as it is not on a compatible grid.")
                continue
            else:
                grid_list.append(data[:, 0])
                current_grid = grid_list[-1]
                values_list.append(data[:, 1])

    grid_array = np.column_stack(grid_list)
    grid_vector = np.unique(grid_array, axis=1)
    values_array = np.column_stack(values_list)
    return grid_vector, values_array


def drawfig(moment_amount, stretching_matrix, weight_matrix, grid_vector):
    plt.ion()
    fig = plt.figure()
    grid = plt.GridSpec(4, 4)
    stretching_plot = fig.add_subplot(grid[0:2, 0:2])
    weight_plot = fig.add_subplot(grid[2:, 0:2])
    component_plot = fig.add_subplot(grid[:, 2:])

    stretching_plot.plot(stretching_matrix.T)
    stretching_plot.set_title("Component Stretching Factors")
    stretching_plot.set_xlabel("Moment")
    stretching_plot.set_ylabel("Stretching Factor")

    weight_plot.plot(weight_matrix.T)
    weight_plot.set_title("Component Weights")
    weight_plot.set_ylabel("Weight")
    weight_plot.set_xlabel("Moment")

    component_plot.plot(grid_vector, component_plot)
    component_plot.set_title("Component Signals")
    component_plot.sex_ylabel("g(r)")
    component_plot.set_xlabel("r")

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.tight_layout()
    plt.show()
