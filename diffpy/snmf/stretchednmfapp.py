import numpy as np
import argparse
from pathlib import Path

from diffpy.snmf.io import load_input_signals, initialize_variables
from diffpy.snmf.subroutines import update_weights_matrix, get_residual_matrix, objective_function


def create_parser():
    parser = argparse.ArgumentParser(
        prog="stretched_nmf",
        description="Stretched Nonnegative Matrix Factorization"
    )

    parser.add_argument('-i', '--input-directory', type=str, default=None,
                        help="Directory containing experimental data. Defaults to current working directory.")
    parser.add_argument('-o', '--output-directory', type=str,
                        help="The directory where the results will be written.  Defaults to '<input_directory>/snmf_results'.")
    parser.add_argument('-t', '--data-type', type=str, choices=['xrd', 'pdf'],
                        help="The type of the experimental data.")
    parser.add_argument('components', type=int,
                        help="The number of component signals for the NMF decomposition. Must be an integer greater than 0.")
    parser.add_argument('-v', '--version', action='version', help='Print the software version number')
    args = parser.parse_args()
    return args


def main():
    args = create_parser()
    if args.input_directory is None:
        args.input_directory = Path.cwd()

    grid, data_input, data_type = load_input_signals(args.input_directory)
    if args.data_type is not None:
        variables = initialize_variables(data_input, args.components, args.data_type)
    else:
        variables = initialize_variables(data_input, args.components, data_type)

    if variables["data_type"] == 'pdf':
        lifted_data = data_input - np.ndarray.min(data_input[:])

    weights_matrix = variables["weight_matrix_guess"]
    component_matrix = variables["component_matrix_guess"]
    stretching_factor_matrix = variables["stretching_matrix_guess"]
    maxiter = 300

    for out_iter in range(maxiter):
        weight_matrix = update_weights_matrix(variables["component_amount"], variables["signal_length"],
                                              stretching_factor_matrix, component_matrix, lifted_data,
                                              variables["moment_amount"], weights_matrix, None)
        residual_matrix = get_residual_matrix(component_matrix, weights_matrix, stretching_factor_matrix, lifted_data,
                                              variables["moment_amount"], variables["component_amount"],
                                              variables["signal_length"])
        fun = objective_function(residual_matrix, stretching_factor_matrix, variables["smoothness"],
                                 variables["smoothness_term"], component_matrix, variables["sparsity"])
        print(fun)


if __name__ == "__main__":
    main()
