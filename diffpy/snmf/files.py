from pathlib import Path
import numpy as np


def processing(file_path=None):
    """Processes a directory of a series of PDF/XRD patterns into a usable format.

    Constructs a 2d array out of a directory of PDF/XRD patterns containing each files dependent variable column in a
    new column. Constructs a 1d array containing the independent variable values.

    Parameters
    ----------
    file_path: str or Path object, optional
      The path to the directory containing the input data. If no path is specified, defaults to the current working
      directory.  Accepts a string or a pathlib.Path object.

    Returns
    -------
    tuple
      The output containing a 2d array containing a PDF/XRD pattern as each of its columns and a 1d array containing the
      independent variable values of the PDF/XRD pattern series.

    """

    if file_path is None:
        directory_path = Path.cwd()
    else:
        directory_path = Path(file_path)

    for pattern_path in directory_path.glob('*'):

        if pattern_path.is_file():
            with pattern_path.open() as in_file:
                data_list = [line.strip().split() for line in in_file]
                values_list = []
                grid_list = []
                values_array = np.empty((len(data_list)))
                grid_points = np.empty((len(data_list)))

                for point in data_list:
                    pass


