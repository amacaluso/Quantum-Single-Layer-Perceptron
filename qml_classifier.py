from Utils_qml import *


x1 = [0.6, 0.5, 0.2, 0.9]; y1 = 1; x = np.array(x1)

# x2 = [3,1]; y2 = 1
# x3 = [1,2]; y3 = -1
# x4 = [1,3]; y4 = -1
#
# X = [x1, x2, x3, x4]; X = [x3, x4, x1, x2]
# Y = [y1, y2, y3, y4]; Y = [y3, y4, y1, y2]
# len(X)
#
# plt.scatter([x[0] for x in X], [x[1] for x in X])
# plt.scatter([x[0] for x in [x1, x2]], [x[1] for x in [x3, x4]])
# plt.show()

import numpy as np
from sklearn import preprocessing

x = x.reshape(1, -1)
X_normalized = preprocessing.normalize(x, norm='l2')
np.linalg.norm(x)

x = np.array([0.53896774, 0.79503606, 0.27826503, 0.0])
np.linalg.norm(x)
ang = get_angles(x)

dev = qml.device('default.qubit', wires=5)

@qml.qnode(dev)
def test(angles=None):

    state_preparation(angles)

    return qml.expval(qml.PauliZ(1))


test(angles=ang)

print("x               : ", x)
print("angles          : ", ang)
print("amplitude vector: ", np.real(dev._state))


def layer(W1):
    qml.Rot(W1[0, 0], W1[0, 1], W1[0, 2], wires=1)
    qml.Rot(W1[1, 0], W1[1, 1], W1[1, 2], wires=2)
    qml.CNOT(wires=[0, 1])

def layer(W2):
    qml.Rot(W2[0, 0], W2[0, 1], W2[0, 2], wires=3)
    qml.Rot(W2[1, 0], W2[1, 1], W2[1, 2], wires=4)
    qml.CNOT(wires=[3, 4])


@qml.qnode(dev)
def circuit(weights, angles=None):
    state_preparation(angles)

    qml.CSWAP()

    layer(W)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(param, x=None):
    var = param
    return circuit(var, x=x)


def cost(var, X, Y):
    predictions = [variational_classifier(var, x=x) for x in X]
    return square_loss(Y, predictions)


opt = qml.GradientDescentOptimizer(stepsize=0.1)

for x in X:
    print( x)

np.random.seed(100)
theta1 = np.array( 2*np.pi*np.random.random_sample() )
theta2 = np.array( 2*np.pi*np.random.random_sample() )
theta3 = np.array( 2*np.pi*np.random.random_sample() )
theta4 = np.array( 2*np.pi*np.random.random_sample() )

init_params = np.array([theta1, theta2, theta3, theta4])


variational_classifier(init_params, x=x)
var = init_params


for it in range(100):

    # Update the weights by one optimizer step
    var = opt.step(lambda v: cost(v, X, Y), var)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(var, x=x)) for x in X]
    acc = accuracy(Y, predictions)

    print("Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(it + 1, cost(var, X, Y), acc))
    # Compute accuracy
    predictions = [np.sign(variational_classifier(var, x=x)) for x in X]
    acc = accuracy(Y, predictions)

    #print("Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(it + 1, cost(var, X, Y), acc))





# print('The expectation value {}'.format(circuit(var = [0, 0, 0])))
# print('The expectation value {}'.format(circuit(var = [np.pi/6, np.pi/2, np.pi/4])))
# print('The expectation value {}'.format(circuit(var = [np.pi, np.pi/3, np.pi/2])))
#
# def objective(var ):
#     if circuit(var) >0.5:
#
#
#     return circuit(var)
#
# np.random.seed(2019)
# theta1 = np.array( 2*np.pi*np.random.random_sample() )
# theta2 = np.array( 2*np.pi*np.random.random_sample() )
# theta3 = np.array( 2*np.pi*np.random.random_sample() )
# init_params = np.array([theta1, theta2, theta3])
# print('Initial objective function value {:.7f} for theta={}'.format(circuit(var = init_params),
#                                                                         init_params))
#
# print( circuit(init_params), init_params )
# # Initilize Gradient Descent Optimizer
# opt = qml.GradientDescentOptimizer(stepsize=0.1)
#
# # set the number of steps
# steps = 10
#
# params = init_params
#
# for i in range(steps):
#     # update the circuit parameters
#     params = opt.step(circuit, params)
#     print('Cost after step {:5d}: {: .7f}'.format(i+1, objective(params)))
#     print(params)
#
# print('Optimized rotation angle: {}'.format(params))
#

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