from Utils import *
import pennylane as qml

dev = qml.device('qiskit.basicaer', wires=2)
# use 'qiskit.ibm' instead to run on hardware

@qml.qnode(dev)
def circuit(x, y, z):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    qml.RZ(z, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval.PauliZ(0)

def cost(x, y, z):
    return (1-circuit(x, y, z))**2

# optimization follows
