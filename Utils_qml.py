import numpy as np
import matplotlib.pyplot as plt

# Qiskit
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ, Aer
from qiskit import BasicAer, execute
from qiskit.tools.visualization import plot_state_city
from qiskit.providers.aer import StatevectorSimulator
from qiskit.tools.visualization import circuit_drawer
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Pennylane
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer



def get_angles(x):

    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def statepreparation(a):
    qml.RY(a[0], wires=1)

    qml.CNOT(wires=[1, 2])
    qml.RY(a[1], wires=2)
    qml.CNOT(wires=[1, 2])
    qml.RY(a[2], wires=2)

    qml.PauliX(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.RY(a[3], wires=2)
    qml.CNOT(wires=[1, 2])
    qml.RY(a[4], wires=2)
    qml.PauliX(wires=1)



def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss



def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss



# x = np.array([0.53896774, 0.79503606, 0.27826503, 0.0])
# ang = get_angles(x)
#
#
# @qml.qnode(dev)
# def test(angles=None):
#
#     statepreparation(angles)
#
#     return qml.expval(qml.PauliZ(0))


# test(angles=ang)
#
# print("x               : ", x)
# print("angles          : ", ang)
# print("amplitude vector: ", np.real(dev._state))


##############################################################################
# Note that the ``default.qubit`` simulator provides a shortcut to
# ``statepreparation`` with the command
# ``qml.QubitStateVector(x, wires=[0, 1])``. However, some devices may not
# support an arbitrary state-preparation routine.
#
# Since we are working with only 2 qubits now, we need to update the layer
# function as well.


def layer(W, wires = None):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=wires[0])
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


##############################################################################
# The variational classifier model and its cost remain essentially the
# same, but we have to reload them with the new state preparation and
# layer functions.
