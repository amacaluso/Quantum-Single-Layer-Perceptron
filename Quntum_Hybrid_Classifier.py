import pennylane as qml
from pennylane import numpy as np, expval, var

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit(var1, var2):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    # layer 1
    qml.RX(var1[0], wires=0)
    qml.RZ(var1[1], wires=1)
    qml.CNOT(wires=(0, 1))

    # layer 2
    qml.RX(var2[0], wires=0)
    qml.RY(var2[1], wires=1)
    # qml.RZ(var2[2], wires=2)

    qml.CNOT(wires=(0, 1))
    return expval(qml.PauliY(0)), var(qml.PauliZ(1))

var1 = np.array([0.54, -0.12])
var2 = np.array([-0.6543, 0.123])

opt = qml.GradientDescentOptimizer(0.1)

def cost(params):
    """Trains the output of the circuit such
    that the parameters for each layer
    result in the expectation <Y> on wire
    0 is a magnitude of 2 different from
    the variance var(PauliZ) on wire 1
    """
    var1 = params[:2]
    var2 = params[1:]
    res = circuit(var1, var2)
    return np.abs(res[0] - res[1] + 2)

params = np.concatenate([var1, var2])

for i in range(100):
    params = opt.step(cost, params)
    print("Cost:", cost(params))

print("Final circuit value:", circuit(params))
print("Final parameters:", params)