import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

dev = qml.device('default.qubit', wires=2)

theta = Parameter('θ')

qc = QuantumCircuit(2)
qc.rz(theta, [0])
qc.rx(theta, [0])
qc.cx(0, 1)

@qml.qnode(dev)
def quantum_circuit_with_loaded_subcircuit(x):
    qml.from_qiskit(qc)({theta: x})
    return qml.expval(qml.PauliZ(0))

angle = np.pi/2
result = quantum_circuit_with_loaded_subcircuit(angle)



dev = qml.device('default.qubit', wires=2)

hadamard_qasm = 'OPENQASM 2.0;' \
                'include "qelib1.inc";' \
                'qreg q[1];' \
                'h q[0];'

apply_hadamard = qml.from_qasm(hadamard_qasm)

@qml.qnode(dev)
def circuit_with_hadamards():
    apply_hadamard(wires=[0])
    apply_hadamard(wires=[1])
    qml.Hadamard(wires=[1])
    return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

result = circuit_with_hadamards()