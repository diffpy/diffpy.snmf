import numpy as np
import snmf_class

X0 = np.loadtxt("input/X0.txt", dtype=float)
MM = np.loadtxt("input/MM.txt", dtype=float)
A0 = np.loadtxt("input/A0.txt", dtype=float)
Y0 = np.loadtxt("input/W0.txt", dtype=float)
N, M = MM.shape

# Convert to DataFrames for display
# df_X = pd.DataFrame(X, columns=[f"Comp_{i+1}" for i in range(X.shape[1])])
# df_Y = pd.DataFrame(Y, columns=[f"Sample_{i+1}" for i in range(Y.shape[1])])
# df_MM = pd.DataFrame(MM, columns=[f"Sample_{i+1}" for i in range(MM.shape[1])])
# df_Y0 = pd.DataFrame(Y0, columns=[f"Sample_{i+1}" for i in range(Y0.shape[1])])

# Print the matrices
"""
print("Feature Matrix (X):\n", df_X, "\n")
print("Coefficient Matrix (Y):\n", df_Y, "\n")
print("Data Matrix (MM):\n", df_MM, "\n")
print("Initial Guess (Y0):\n", df_Y0, "\n")
"""

my_model = snmf_class.SNMFOptimizer(MM=MM, Y0=Y0, X0=X0, A=A0, components=2)
print("Done")
# print(f"My final guess for X: {my_model.X}")
# print(f"My final guess for Y: {my_model.Y}")
# print(f"Compare to true X: {X_norm}")
# print(f"Compare to true Y: {Y_norm}")
np.savetxt("my_norm_X.txt", my_model.X, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_Y.txt", my_model.Y, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_A.txt", my_model.A, fmt="%.6g", delimiter=" ")
