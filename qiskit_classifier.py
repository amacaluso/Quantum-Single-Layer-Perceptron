import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ, Aer
from qiskit import BasicAer, execute
from qiskit.tools.visualization import plot_state_city
from qiskit.providers.aer import StatevectorSimulator
from qiskit.tools.visualization import circuit_drawer
import numpy as np
from qiskit.circuit import Parameter


def normalize_custom(x, C =1):
    M = x[0] ** 2 + x[1] ** 2

    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed

theta = Parameter('θ')

''' State Preparation'''
## ++++++++++++++++++++++++++++++++++ ##
y_class0 = [1, 0]
y_class1 = [0, 1]

# Training Set
x1_train = [1, 3]; x1_train = normalize_custom(x1_train) # y1 = '0'
x2_train = [3, -1]; x2_train = normalize_custom(x2_train) # y2 = '1'
x3_train = [1.5, 2.5]; x3_train = normalize_custom(x3_train)# y3 = '0'
x4_train = [2, -2]; x4_train = normalize_custom(x4_train)# y4 = '1'
D = [x2_train, x1_train, x4_train, x3_train]


''' Quantum Circuit for Ensemble '''
# dim(D) and label y1, y2, y3, y4 fixed
''' Create Circuit '''
# Create a Classical Register with 1 bit.
c = ClassicalRegister(1)
# Create a Quantum Circuit
ancilla = QuantumRegister(1)
phi = QuantumRegister(4, 'phi')
psi = QuantumRegister(4, 'psi')
qc = QuantumCircuit(ancilla, psi, phi, c)

ancilla = ancilla[0]

x1 = psi[0]
x2 = psi[1]
y1 = psi[2]
y2 = psi[3]

### Initialization ###

qc.initialize(x1_train, [x1])
# qc.barrier()
qc.initialize(x2_train, [x2])
qc.initialize(y_class1, [y1])
qc.initialize(y_class0, [y2])

qc.barrier()
qc.cswap(ancilla, psi, phi)
qc.barrier()

## +++++++++++++++++++ ##
# U1
qc.rx(theta, x1)

# U2
qc.swap(phi[1], phi[3])
## +++++++++++++++++++ ##

qc.barrier()
qc.cswap(ancilla1, phi, psi)
qc.barrier()

qc.cswap(ancilla2, phi, psi)
qc.barrier()

## +++++++++++++++++++ ##
# U3
qc.swap(x3, x_train)

# U4
qc.swap(phi[1], phi[4])
qc.swap(phi[7], phi[8])

## +++++++++++++++++++ ##
qc.barrier()
qc.cswap(ancilla2, phi, psi)
qc.barrier()

# C
qc.h(yt)
qc.cswap(yt, xt, x_train)
qc.h(yt)
qc.cx(y_train, yt)
qc.measure(yt, c)
# print(qc)