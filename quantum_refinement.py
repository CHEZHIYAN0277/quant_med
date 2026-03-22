from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC


def quantum_refinement(X, y):

    feature_map = ZZFeatureMap(
        feature_dimension=X.shape[1],
        reps=1
    )

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    kernel_matrix = quantum_kernel.evaluate(X)

    model = SVC(kernel="precomputed")

    model.fit(kernel_matrix, y)

    scores = model.decision_function(kernel_matrix)

    return scores