import os.path, sys
dir = os.path.join('qiskit implementation')
sys.path.insert(0, dir)


from Utils import *


def state_preparation(x):
    backend = Aer.get_backend('unitary_simulator')

    x = normalize_custom(x)

    qreg = QuantumRegister(1)
    qc = QuantumCircuit(qreg)
    # Run the quantum circuit on a unitary simulator backend
    qc.initialize(x, [qreg])
    job = execute(qc, backend)
    result = job.result()

    U = result.get_unitary(qc)
    S = Operator(U)
    return S


def predict(probas):
    return (probas >= 0.5) * 1


def binary_crossentropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        # print(l,p)
        loss = loss - l * np.log(np.max([p, 1e-8]))

    loss = loss / len(labels)
    return loss


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



def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])



def state_preparation(a, circuit, target):
    a = 2 * a
    circuit.ry(a[0], target[0])

    circuit.cx(target[0], target[1])
    circuit.ry(a[1], target[1])
    circuit.cx(target[0], target[1])
    circuit.ry(a[2], target[1])

    circuit.x(target[0])
    circuit.cx(target[0], target[1])
    circuit.ry(a[3], target[1])
    circuit.cx(target[0], target[1])
    circuit.ry(a[4], target[1])
    circuit.x(target[0])

    return circuit