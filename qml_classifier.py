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


### Initialization ###

import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.qubit', wires=3)


@qml.qnode(dev)
def circuit(var):
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.Hadamard(2)
    qml.CSWAP(wires=[0,1,2])
    # qml.CSWAP(wires=[0, 2, 4])
    qml.RY(var[0], wires=1)
    qml.RY(var[1], wires=2)
    qml.CSWAP(wires=[0,1,2])
    qml.RY(var[2], wires=1)
    return qml.expval(qml.PauliZ(1))

print('The expectation value {}'.format(circuit(var = [0, 0, 0])))
print('The expectation value {}'.format(circuit(var = [np.pi/6, np.pi/2, np.pi/4])))
print('The expectation value {}'.format(circuit(var = [np.pi, np.pi/3, np.pi/2])))

def objective(var ):
    return circuit(var)

np.random.seed(2019)
theta1 = np.array( 2*np.pi*np.random.random_sample() )
theta2 = np.array( 2*np.pi*np.random.random_sample() )
theta3 = np.array( 2*np.pi*np.random.random_sample() )
init_params = np.array([theta1, theta2, theta3])
print('Initial objective function value {:.7f} for theta={}'.format(circuit(var = init_params),
                                                                        init_params))

print( circuit(init_params), init_params )
# Initilize Gradient Descent Optimizer
opt = qml.GradientDescentOptimizer()

# set the number of steps
steps = 1000

params = init_params

for i in range(steps):
    # update the circuit parameters
    params = opt.step(circuit, params)
    print('Cost after step {:5d}: {: .7f}'.format(i+1, objective(params)))
    print(params)

print('Optimized rotation angle: {}'.format(params))


# import pennylane as qml
# from pennylane import numpy as np
#
# dev = qml.device('default.qubit', wires=1)
#
# @qml.qnode(dev)
# def circuit(params):
#     qml.RY(params[0], wires=0)
#     return qml.expval(qml.PauliZ(0))
#
# print('The expectation value {}'.format(circuit([0])))
# print('The expectation value {}'.format(circuit([np.pi/3])))
# print('The expectation value {}'.format(circuit([np.pi])))
#
# def objective(var):
#     return circuit(var)
#
# np.random.seed(2019)
# initial_theta = 2*np.pi*np.random.random_sample()
# init_params = np.array([initial_theta])
# print('Initial objective function value {:.7f} for theta={:.2f}'.format(objective(init_params),
#                                                                         initial_theta))
# # Initilize Gradient Descent Optimizer
# opt = qml.GradientDescentOptimizer(stepsize=0.4)
#
# # set the number of steps
# steps = 30
# # set the initial parameter values
# params = init_params
#
# for i in range(steps):
#     # update the circuit parameters
#     params = opt.step(objective, params)
#     print('Cost after step {:5d}: {: .7f}'.format(i+1, objective(params)))
#
# print('Optimized rotation angle: {}'.format(params))