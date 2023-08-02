import numpy as np
import argparse
from pathlib import Path

ALLOWED_DATA_TYPES = ['powder_diffraction', 'pd', 'pair_distribution_function', 'pdf']


def create_parser():
    parser = argparse.ArgumentParser(
        prog="stretched_nmf",
        description="Stretched Nonnegative Matrix Factorization"
    )
    parser.add_argument('-i', '--input-directory', type=str, default=None,
                        help="Directory containing experimental data. Defaults to current working directory.")
    parser.add_argument('-o', '--output-directory', type=str,
                        help="The directory where the results will be written. Defaults to '<input_directory/snmf_results>'.")
    parser.add_argument('t', '--data-type', type=str, choices=ALLOWED_DATA_TYPES,
                        help="The type of the experimental data.")
    parser.add_argument('-l', '--lift-factor', type=float, default=1,
                        help="The lifting factor. Data will be lifted by lifted_data = data + abs(min(data) * lift). Default is 1.")
    parser.add_argument('number-of-components', type=int,
                        help="The number of component signals for the NMF decomposition. Must be an integer greater than 0")
    parser.add_argument('-v', '--version', action='version', help='Print the software version number')
    args = parser.parse_args()
    return args


def main():
    pass
