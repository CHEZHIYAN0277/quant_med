from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.svm import SVC


def quantum_refinement(X, y):

    feature_map = ZZFeatureMap(
        feature_dimension=3,
        reps=1
    )

    backend = Aer.get_backend("aer_simulator_statevector")

    quantum_instance = QuantumInstance(backend)

    quantum_kernel = QuantumKernel(
        feature_map=feature_map,
        quantum_instance=quantum_instance
    )

    kernel_matrix = quantum_kernel.evaluate(X)

    model = SVC(kernel="precomputed")

    model.fit(kernel_matrix, y)

    scores = model.decision_function(kernel_matrix)

    return scores