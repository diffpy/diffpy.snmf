import numpy as np

from diffpy.snmf.io import load_input_signals, initialize_variables


def main():
    directory_path = input("Specify Path (Optional. Press enter to skip):")
    if not directory_path:
        directory_path = None

    data_type = input("Specify the data type ('xrd' or 'pdf'): ")
    if data_type != 'xrd' and data_type != 'pdf':
        raise ValueError("The data type must be 'xrd' or 'pdf'")

    component_amount = input("\nEnter the amount of components to obtain:")
    try:
        component_amount = int(component_amount)
    except TypeError:
        raise TypeError("Please enter an integer greater than 0")

    grid, data_input = load_input_signals(directory_path)
    variables = initialize_variables(data_input, component_amount, data_type)
    lifted_data = data_input - np.ndarray.min(data_input[:])


if __name__ == "__main__":
    main()
