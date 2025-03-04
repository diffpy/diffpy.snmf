import numpy as np
import pandas as pd
import snmf

# Define fixed feature matrix (X) with distinct, structured features
X = np.array(
    [
        [10, 0, 0],  # First component dominates first feature
        [0, 8, 0],  # Second component dominates second feature
        [0, 0, 6],  # Third component dominates third feature
        [4, 4, 0],  # Mixed contribution to the fourth feature
        [3, 2, 5],  # Mixed contribution to the fifth feature
    ],
    dtype=float,
)

# Define fixed coefficient matrix (Y) representing weights
Y = np.array(
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]],
    dtype=float,
)

# Compute the resulting data matrix M
MM = np.dot(X, Y)

# Normalize matrices X, Y, and M to be between 0 and 1
X_norm = (X - X.min()) / (X.max() - X.min())
Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
MM_norm = (MM - MM.min()) / (MM.max() - MM.min())

# Generate an initial guess Y0 with slightly perturbed values
Y0 = np.array(
    [
        [1.5, 1.8, 2.9, 3.6, 4.8, 5.7, 7.1, 8.2, 9.4, 10.3],
        [2.2, 4.1, 5.9, 8.1, 9.8, 11.9, 14.2, 16.5, 18.1, 19.7],
        [2.7, 5.5, 8.8, 11.5, 14.6, 17.8, 20.5, 23.9, 26.3, 29.2],
    ],
    dtype=float,
)

# Normalize Y0 as well
Y0_norm = (Y0 - Y0.min()) / (Y0.max() - Y0.min())

# Convert to DataFrames for display
df_X = pd.DataFrame(X, columns=[f"Comp_{i+1}" for i in range(X.shape[1])])
df_Y = pd.DataFrame(Y, columns=[f"Sample_{i+1}" for i in range(Y.shape[1])])
df_MM = pd.DataFrame(MM, columns=[f"Sample_{i+1}" for i in range(MM.shape[1])])
df_Y0 = pd.DataFrame(Y0, columns=[f"Sample_{i+1}" for i in range(Y0.shape[1])])

# Print the matrices
"""
print("Feature Matrix (X):\n", df_X, "\n")
print("Coefficient Matrix (Y):\n", df_Y, "\n")
print("Data Matrix (MM):\n", df_MM, "\n")
print("Initial Guess (Y0):\n", df_Y0, "\n")
"""

my_model = snmf.SNMFOptimizer(MM_norm, Y0_norm)
print(f"My final guess for X: {my_model.X}")
print(f"My final guess for Y: {my_model.Y}")
print(f"Compare to true X: {X_norm}")
print(f"Compare to true Y: {Y_norm}")
