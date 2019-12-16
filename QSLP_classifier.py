from Utils_qml import *
from pennylane.templates import AmplitudeEmbedding

x0 = np.array([0.0, 0.1])
x1 = np.array([0.5, 0.1])
x2 = np.array([0.9, 0.7])
x3 = np.array([0.0, 0.0])
X = [ x0, x1, x2, x3]

y = [1, -1, -1, 1]


dev = qml.device('default.qubit', wires=1)


def state_preparation(x=None):
    AmplitudeEmbedding (x, wires=[0], normalize=True)



# @qml.qnode(dev)
# def test(x=None):
#     state_preparation(x)
#     return qml.expval(qml.PauliZ(0))

# test(x=x)
#
# print("x               : ", x)
# print("amplitude vector: ", np.real(dev._state))

def layer(W):
    qml.Rot(W[0], W[1], W[2], wires=0)



@qml.qnode(dev)
def circuit(weights, vector=None):
    state_preparation(vector)
    layer(weights)
    return qml.expval(qml.PauliZ(0))


def variational_classifier(var, data=None):
    weights = var[0]
    bias = var[1]
    return circuit(weights, vector=data) + bias


def cost(weights, data, labels):
    predictions = [variational_classifier(weights, data=data)]
    return square_loss(labels, predictions)

# predictions = [variational_classifier(var, data=x)]
# square_loss(y, predictions[0])

np.random.seed(100)
theta = [ 2*np.pi*np.random.random_sample(),
          2*np.pi*np.random.random_sample(),
          2*np.pi*np.random.random_sample()]

init_params = (theta, 0.0)


variational_classifier(init_params, data = x)
var = init_params

from pennylane.optimize import NesterovMomentumOptimizer
opt = NesterovMomentumOptimizer(0.01)


for it in range(10):

    # Update the weights by one optimizer step
    var = opt.step(lambda v: cost(var, X, y), var)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(var, data=x)) for x in X]
    acc = accuracy(y, predictions)

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