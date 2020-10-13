# Copyright 2020 Antonio Macaluso
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from Utils import *


# Data
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=500, centers=[[0.2, 0.8],[0.7, 0.1]],
                           n_features=2, center_box=(0, 1),
                           cluster_std = 0.2, random_state = 5432)
Y = np.where(y == 0, -1, 1)

# pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]

# normalize each input
normalization = np.sqrt(np.sum(X_pad ** 2, -1))
X_norm = (X_pad.T / normalization).T

# angles for state preparation are new features
features = np.array([get_angles(x) for x in X_norm])


dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def circuit(weights, angles=None):
    """
        Generete the circuit of the two-neurons qSLP
        :param weights: (float) input parameters of the circuit
        :param angles: (float) rotation to encode data in the quantum register
        :return: (obj) quantum circuit
        """
    statepreparation(angles)
    qml.RY(weights[2], wires=0)
    qml.CSWAP(wires=[0, 1, 3])
    qml.CSWAP(wires = [0, 2, 4])

    for W in weights[0]:
        layer(W, wires = [1,2])

    for W in weights[1]:
        layer(W, [3,4])

    qml.CSWAP(wires = [0, 1, 3])
    qml.CSWAP(wires = [0, 2, 4])
    # qml.RY(weights[2], wires=1)
    return qml.expval(qml.PauliZ(1))


def variational_classifier(var, angles=None):
    """
    Define the variational classifier and its parameter
    :param var: parameter of the quantum gates
    :param angles: (float) rotation to encode data in the quantum register
    :return: (float) the prediction of the quantum classifier given  the angles encoding the input
    """
    weights = var[0]
    bias = var[1]
    return circuit(weights, angles=angles) + bias


def cost(weights, features, labels):
    """
    Compute the cost function of the classifier
    :param weights: (float) vector of parameters of the quantum circuit
    :param features: (float) vector of features in terms of angles for state preparation
    :param labels: (integer) target variable to approximate
    :return: (float) value of the computed cost function
    """
    predictions = [variational_classifier(weights, angles=f) for f in features]
    return square_loss(labels, predictions)



# Splitting data in training and validation to monitor the performance

np.random.seed(0)
num_data = len(Y)
num_train = int(0.75 * num_data)
index = np.random.permutation(range(num_data))
feats_train = features[index[:num_train]]
Y_train = Y[index[:num_train]]
feats_val = features[index[num_train:]]
Y_val = Y[index[num_train:]]

# We need these later for plotting
X_train = X[index[:num_train]]
X_val = X[index[num_train:]]

#### Optimisation

# Optimiser
opt = NesterovMomentumOptimizer(0.01)
batch_size = 10 # num_train

# train the variational classifier
acc_final_tr = 0
acc_final_val = 0
num_qubits = 2
num_layers = 1
cost_vector = []
train_vector = []
val_vector = []

seeds =[000]

for seed in seeds:
    np.random.seed(seed)
    var_init = ((0.01 * np.random.randn(num_layers, num_qubits, 3),
                 0.01 * np.random.randn(num_layers, num_qubits, 3),
                 2*np.pi*np.random.random_sample()),
                0.0)
    var = var_init

    best_param = var_init
    for it in range(100):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, num_train, (batch_size,))
        feats_train_batch = feats_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        var = opt.step(lambda v: cost(v, feats_train_batch, Y_train_batch), var)

        # Compute predictions on train and validation set
        predictions_train = [np.sign(variational_classifier(var, angles=f)) for f in feats_train]
        predictions_val = [np.sign(variational_classifier(var, angles=f)) for f in feats_val]

        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)
        if acc_final_tr < acc_train:
            best_param = var
            acc_final_tr = acc_train
            acc_final_val = acc_val
            best_seed = seed
            iteration = it
        cost_vector.append(cost(var, features, Y))
        train_vector.append(acc_train)
        val_vector.append(acc_val)

        print(
            "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
            "".format(it + 1, cost(var, features, Y), acc_train, acc_val))

