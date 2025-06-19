import numpy as np
from snmf_class import SNMFOptimizer

init_components_file = np.loadtxt("input/X0.txt", dtype=float)
source_matrix_file = np.loadtxt("input/MM.txt", dtype=float)
init_stretch_file = np.loadtxt("input/A0.txt", dtype=float)
init_weights_file = np.loadtxt("input/W0.txt", dtype=float)

my_model = SNMFOptimizer(
    source_matrix=source_matrix_file,
    init_weights=init_weights_file,
    init_components=init_components_file,
    init_stretch=init_stretch_file,
)

print("Done")
np.savetxt("my_norm_components.txt", my_model.components, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_weights.txt", my_model.weights, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_stretch.txt", my_model.stretch, fmt="%.6g", delimiter=" ")
