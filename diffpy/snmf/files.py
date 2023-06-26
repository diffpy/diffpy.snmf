import os
import numpy as np

def processing(*path):
    """
    Returns an ndarray of PDF/XRD values and a vector of independent variable values corresponding with each output.

    Parameters
    ----------
    path: string
       An optional string specifying the location of a file containing only PDF/XRD patterns. If empty, the program will
       look in the current directory.

    Returns
    -------

    """

    pattern_list = []
    directory = os.fsencode(*path)
    for file in os.scandir(directory):
        pattern_list.append(file)

    for file in pattern_list:
        with open(fr"{file}", 'r') as input_file: