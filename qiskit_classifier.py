import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ, Aer
from qiskit import BasicAer, execute
from qiskit.tools.visualization import plot_state_city
from qiskit.providers.aer import StatevectorSimulator
from qiskit.tools.visualization import circuit_drawer
import numpy as np
from qiskit.circuit import Parameter
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

theta = Parameter('A')
# gamma = Parameter('B')

dev = qml.device('strawberryfields.fock', wires=5, cutoff_dim=10)



def normalize_custom(x, C =1):
    M = x[0] ** 2 + x[1] ** 2

    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed


''' State Preparation'''
## ++++++++++++++++++++++++++++++++++ ##
y_class0 = [1, 0]
y_class1 = [0, 1]

# Training Set
x1_train = [1, 3]; x1_train = normalize_custom(x1_train) # y1 = '0'
x2_train = [3, -1]; x2_train = normalize_custom(x2_train) # y2 = '1'
# x3_train = [1.5, 2.5]; x3_train = normalize_custom(x3_train)# y3 = '0'
# x4_train = [2, -2]; x4_train = normalize_custom(x4_train)# y4 = '1'
# D = [x2_train, x1_train, x4_train, x3_train]


''' Quantum Circuit for Ensemble '''
# dim(D) and label y1, y2, y3, y4 fixed
''' Create Circuit '''
# Create a Classical Register with 1 bit.
#c = ClassicalRegister(1)
# Create a Quantum Circuit
ancilla = QuantumRegister(1)
phi = QuantumRegister(2, 'data')
psi = QuantumRegister(2, 'temp')
qc = QuantumCircuit(ancilla, psi, phi)

#ancilla = ancilla[0]


x1 = psi[0]
x2 = psi[1]

### Initialization ###

qc.initialize(x1_train, [x1])
qc.initialize(x2_train, [x2])
# qc.initialize(y_class1, [y1])
# qc.initialize(y_class0, [y2])

qc.h(ancilla)
qc.h(phi)

qc.barrier()
qc.cswap(ancilla, x1, phi[0])
qc.cswap(ancilla, x2, phi[1])
qc.barrier()

## +++++++++++++++++++ ##
# U1
qc.rx(theta, x1)

# U2
#qc.rx(gamma, x2)
## +++++++++++++++++++ ##

qc.barrier()
qc.cswap(ancilla, x1, phi[0])
qc.cswap(ancilla, x2, phi[1])
qc.barrier()
## +++++++++++++++++++ ##

# C
## +++++++++++++++++++ ##
# U1
# qc.rx(theta, x1)
print(qc)


@qml.qnode(dev)
def quantum_circuit(x): #, y):
    qml.from_qiskit(qc)({theta: x}) #, gamma: y})
    return qml.expval(qml.PauliX(1))

angle = np.pi/4
result = quantum_circuit(angle/4)

# def cost(params, phi1=0.5, phi2=0.1):
#     """Returns the squared difference between
#     the photon-redirection and qubit-rotation QNodes, for
#     fixed values of the qubit rotation angles phi1 and phi2"""
#     qubit_result = qubit_rotation(phi1, phi2)
#     photon_result = photon_redirection(params)
#     return squared_difference(qubit_result, photon_result)
#
