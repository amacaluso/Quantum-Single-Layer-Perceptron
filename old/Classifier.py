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

print(qc)

@qml.qnode(dev)
def quantum_circuit_with_loaded_subcircuit(x):
    qml.from_qiskit(qc)({theta: x})
    return qml.expval(qml.PauliZ(0))

angle = np.pi/2
result = quantum_circuit_with_loaded_subcircuit(angle)

def cost(params, phi1=0.5, phi2=0.1):
    """Returns the squared difference between
    the photon-redirection and qubit-rotation QNodes, for
    fixed values of the qubit rotation angles phi1 and phi2"""
    qubit_result = qubit_rotation(phi1, phi2)
    photon_result = photon_redirection(params)
    return squared_difference(qubit_result, photon_result)



# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 100
# set the initial parameter values
params = np.array([0.01, 0.01])

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)

    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

print("Optimized rotation angles: {}".format(params))



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