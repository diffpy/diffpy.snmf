import numpy as np
from snmf_class import SNMFOptimizer

init_comps_file = np.loadtxt("input/X0.txt", dtype=float)
source_matrix_file = np.loadtxt("input/MM.txt", dtype=float)
init_stretch_file = np.loadtxt("input/A0.txt", dtype=float)
init_weights_file = np.loadtxt("input/W0.txt", dtype=float)

my_model = SNMFOptimizer(
    source_matrix=source_matrix_file,
    init_weights=init_weights_file,
    init_comps=init_comps_file,
    init_stretch=init_stretch_file,
)

print("Done")
np.savetxt("my_norm_comps.txt", my_model.comps, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_weights.txt", my_model.weights, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_stretch.txt", my_model.stretch, fmt="%.6g", delimiter=" ")
