import numpy as np
import argparse

from diffpy.snmf.io import load_input_signals, initialize_variables


def create_parser():
    parser = argparse.ArgumentParser(
        prog="stretched_nmf",
        description="Stretched Nonnegative Matrix Factorization"
    )
    parser.add_argument('-v', '--version', action='version', help='Print the software version number')
    parser.add_argument('-d', '--directory', type=str,
                        help="Directory containing experimental data. Ensure it is in quotations or apostrophes.")

    parser.add_argument('component_number', type=int,
                        help="The number of component signals to obtain from experimental "
                             "data. Must be an integer greater than 0.")
    parser.add_argument('data_type', type=str, choices=['xrd', 'pdf'], help="The type of the experimental data.")
    args = parser.parse_args()
    return args


def main():
    args = create_parser()

    grid, data_input = load_input_signals(args.directory)
    variables = initialize_variables(data_input, args.component_number, args.data_type)
    if variables["data_type"] == 'pdf':
        lifted_data = data_input - np.ndarray.min(data_input[:])
    maxiter = 300
    return lifted_data


if __name__ == "__main__":
    main()
