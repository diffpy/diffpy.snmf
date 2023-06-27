from pathlib import Path
import numpy as np


def processing(*path):
    """Processes a directory of a series of PDF/XRD patterns into a usable format.

    Constructs a 2d array out of a directory of PDF/XRD patterns containing each files dependent variable column in a
    new column. Constructs a 1d array containing the independent variable values.

    Parameters
    ----------
    path: str, optional
      The specific location/path of a directory containing only PDF or XRD data. The default is the path of the
      current working directory. It may be a POSIX or Windows path.

    Returns
    -------
    tuple
      The output containing a 2d array containing a PDF/XRD pattern as each of its columns and a 1d array containing the
      independent variable values of the PDF/XRD pattern series.

    """
    if not path:
        directory_path = Path.cwd()
    else:
        directory_path = Path(path)

    for pattern_path in directory_path.glob('*'):
        if pattern_path.is_file():
            with pattern_path.open() as in_file:
                for line in in_file: