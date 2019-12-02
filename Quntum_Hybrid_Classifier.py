import pennylane as qml
from pennylane import numpy as np

##############################################################################
# Next, we create a device to run the quantum node. This is easy in PennyLane; as soon as
# the PennyLane-SF plugin is installed, the ``'strawberryfields.fock'`` device can be loaded
# — no additional commands or library imports required.

dev_fock = qml.device("strawberryfields.fock", wires=2, cutoff_dim=2)


##############################################################################
# Constructing the QNode
# ----------------------
#
# Now that we have initialized the device, we can construct our quantum node. Like
# the other tutorials, we use the :mod:`~.pennylane.qnode` decorator
# to convert our quantum function (encoded by the circuit above) into a quantum node
# running on Strawberry Fields.


@qml.qnode(dev_fock)
def photon_redirection(params):
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0, 1])
    return qml.expval(qml.NumberOperator(1))


##############################################################################
# Optimization
# ------------
#
# Let's now use one of the built-in PennyLane optimizers in order to
# carry out photon redirection. Since we wish to maximize the mean photon number of
# the second wire, we can define our cost function to minimize the *negative* of the circuit output.


def cost(params):
    return -photon_redirection(params)


##############################################################################
# To begin our optimization, let's choose the following small initial values of
# :math:`\theta` and :math:`\phi`:

init_params = np.array([0.01, 0.01])
print(cost(init_params))

##############################################################################
# Here, we choose the values of :math:`\theta` and :math:`\phi` to be very close to zero;
# this results in :math:`B(\theta,\phi)\approx I`, and the output of the quantum
# circuit will be very close to :math:`\left|1, 0\right\rangle` — i.e., the circuit leaves the photon in the first mode.
#
# Why don't we choose :math:`\theta=0` and :math:`\phi=0`?
#
# At this point in the parameter space, :math:`\left\langle \hat{n}_1\right\rangle = 0`, and
# :math:`\frac{d}{d\theta}\left\langle{\hat{n}_1}\right\rangle|_{\theta=0}=2\sin\theta\cos\theta|_{\theta=0}=0`.
# Since the gradient is zero at those initial parameter values, the optimization
# algorithm would never descend from the maximum.
#
# This can also be verified directly using PennyLane:

dphoton_redirection = qml.grad(photon_redirection, argnum=0)
print(dphoton_redirection([0.0, 0.0]))

##############################################################################
# Now, let's use the :class:`~.pennylane.GradientDescentOptimizer`, and update the circuit
# parameters over 100 optimization steps.

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 100
# set the initial parameter values
params = init_params

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)

    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

print("Optimized rotation angles: {}".format(params))


##############################################################################
# Comparing this to the :ref:`exact calculation <photon_redirection_calc>` above,
# this is close to the optimum value of :math:`\theta=\pi/2`, while the value of
# :math:`\phi` has not changed — consistent with the fact that :math:`\left\langle \hat{n}_1\right\rangle`
# is independent of :math:`\phi`.
#
# .. _hybrid_computation_example:
#
# Hybrid computation
# ------------------
#
# To really highlight the capabilities of PennyLane, let's now combine the qubit-rotation QNode
# from the :ref:`qubit rotation tutorial <qubit_rotation>` with the CV photon-redirection
# QNode from above, as well as some classical processing, to produce a truly hybrid
# computational model.
#
# First, we define a computation consisting of three steps: two quantum nodes (the qubit rotation
# and photon redirection circuits, running on the ``'default.qubit'`` and
# ``'strawberryfields.fock'`` devices, respectively), along with a classical function, that simply
# returns the squared difference of its two inputs using NumPy:

# create the devices
dev_qubit = qml.device("default.qubit", wires=1)
dev_fock = qml.device("strawberryfields.fock", wires=2, cutoff_dim=10)


@qml.qnode(dev_qubit)
def qubit_rotation(phi1, phi2):
    """Qubit rotation QNode"""
    qml.RX(phi1, wires=0)
    qml.RY(phi2, wires=0)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_fock)
def photon_redirection(params):
    """The photon redirection QNode"""
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0, 1])
    return qml.expval(qml.NumberOperator(1))


def squared_difference(x, y):
    """Classical node to compute the squared
    difference between two inputs"""
    return np.abs(x - y) ** 2




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

##############################################################################
# Substituting this into the photon redirection QNode shows that it now produces
# the same output as the qubit rotation QNode:

result = [1.20671364, 0.01]
print(photon_redirection(result))
print(qubit_rotation(0.5, 0.1))

##############################################################################
# This is just a simple example of the kind of hybrid computation that can be carried
# out in PennyLane. Quantum nodes (bound to different devices) and classical
# functions can be combined in many different and interesting ways.
