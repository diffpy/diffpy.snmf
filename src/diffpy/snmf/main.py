import numpy as np
import snmf_class

X0 = np.loadtxt("input/X0.txt", dtype=float)
MM = np.loadtxt("input/MM.txt", dtype=float)
A0 = np.loadtxt("input/A0.txt", dtype=float)
Y0 = np.loadtxt("input/W0.txt", dtype=float)
N, M = MM.shape

my_model = snmf_class.SNMFOptimizer(MM=MM, Y0=Y0, X0=X0, A=A0, n_components=2)
print("Done")
np.savetxt("my_norm_X.txt", my_model.X, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_Y.txt", my_model.Y, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_A.txt", my_model.A, fmt="%.6g", delimiter=" ")
