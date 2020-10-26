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


from qiskit_Utils import *
from import_data import *

X,Y = load_parity()
dataset = 'parity'

# X,Y = load_moon()
dataset = 'moon'

X,Y = load_bivariate_gaussian()
dataset = 'gaussian'


# X,Y = load_iris(type=1)
# X,Y = load_iris(type=0)


# pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
print("First X sample (padded)    :", X_pad[0])

# normalize each input
normalization = np.sqrt(np.sum(X_pad ** 2, -1))
X_norm = (X_pad.T / normalization).T
print("First X sample (normalized):", X_norm[0])

# angles for state preparation are new features
features = np.nan_to_num((np.array([get_angles(x) for x in X_norm])))
print("First features sample      :", features[0])

def get_Sx(ang):
    backend = Aer.get_backend('unitary_simulator')

    q = QuantumRegister(2)
    circuit = QuantumCircuit(q)
    circuit = state_preparation(ang, circuit, [0, 1])

    job = execute(circuit, backend)
    result = job.result()

    U = result.get_unitary(circuit)
    S = Operator(U)
    return S


def linear_operator(param):
    backend = Aer.get_backend('unitary_simulator')

    data_reg = QuantumRegister(2)
    qc = QuantumCircuit(data_reg)
    qc.u(param[0], param[1], param[2], data_reg[0])
    qc.u(param[3], param[4], param[5], data_reg[1])
    qc.cx(data_reg[0], data_reg[1])

    job = execute(qc, backend)
    result = job.result()

    U = result.get_unitary(qc)
    G = Operator(U)
    return G

def sigma():
    backend = Aer.get_backend('unitary_simulator')
    data = QuantumRegister(2)
    qc = QuantumCircuit(data)
    qc.id(data)

    job = execute(qc, backend)
    result = job.result()

    U = result.get_unitary(qc)
    I = Operator(U)
    return I

def R_gate(beta):
    backend = Aer.get_backend('unitary_simulator')
    control = QuantumRegister(1)
    qc = QuantumCircuit(control)
    qc.ry(beta, control)

    job = execute(qc, backend)
    result = job.result()

    U = result.get_unitary(qc)
    R = Operator(U)
    return R

def execute_circuit(parameters, x=None, shots=1000, print=False):
    '''
    :param parameters:
    :param x:
    :param shots:
    :param print:
    :return:
    '''

    # Define the circuit for the quantum Single Layer Perceptron
    beta = parameters[0]
    theta1 = parameters[1:7]
    theta2 = parameters[7:13]

    control = QuantumRegister(1, 'control')
    data = QuantumRegister(2, 'x')
    temp = QuantumRegister(2, 'temp')
    c = ClassicalRegister(1)
    qc = QuantumCircuit(control, data, temp, c)

    ang = np.nan_to_num(get_angles(x))
    S = get_Sx(ang)
    qc.unitary(S, data, label='$S_{x}$')

    R = R_gate(beta)
    qc.unitary(R, control, label='$R_{Y}(β)$')

    qc.barrier()
    qc.cswap(control, data[0], temp[0])
    qc.cswap(control, data[1], temp[1])

    G1 = linear_operator(theta1)
    qc.unitary(G1, data, label='$G(θ_{1})$')


    G2 = linear_operator(theta2)
    qc.unitary(G2, temp, label='$G(θ_{2})$')

    qc.cswap(control, data[1], temp[1])
    qc.cswap(control, data[0], temp[0])

    sig = sigma()
    qc.unitary(sig, data, label='$Σ$')

    qc.barrier()
    qc.measure(data[0], c)

    # Execute the qSLP
    backend = BasicAer.get_backend('qasm_simulator')
    if print:
        qc.draw(output='mpl')
        plt.show()
    result = execute(qc, backend, shots=shots).result()

    counts = result.get_counts(qc)
    result = np.zeros(2)
    for key in counts:
        result[int(key, 2)] = counts[key]
    result /= shots
    return result[1]



def cost(params, X, labels):
    predictions = [execute_circuit(params, x) for x in X]
    return binary_crossentropy(labels, predictions)


X = X_norm.copy()
# seed = 974 # iris:359, gaussian:527
seed=np.random.randint(0,10**3,1)[0]
np.random.seed(seed)
point = (0.1*np.random.randn(13))

from qiskit.aqua.components.optimizers import AQGD
optimizer_step = AQGD(maxiter=1, eta=2.0, disp=False)
execute_circuit(point, x=X[2], print=True)


num_data = len(Y)
num_train = int(0.75 * num_data)
index = np.random.permutation(range(num_data))
X_train = X[index[:num_train]]
Y_train = Y[index[:num_train]]
X_val = X[index[num_train:]]
Y_val = Y[index[num_train:]]
batch_size = 10
T = 10
acc_final_tr = 0
acc_final_val = 0

for i in range(T):
    batch_index = np.random.randint(0, num_train, (batch_size,))
    X_batch = X_train[batch_index]
    Y_batch = Y_train[batch_index]

    obj_function = lambda params: cost(params, X_batch, Y_batch)
    point, value, fev = optimizer_step.optimize(len(point), obj_function,
                                                initial_point=point)

    # Compute predictions on train and validation set
    probs_train = [execute_circuit(point, x) for x in X_train]
    probs_val = [execute_circuit(point, x) for x in X_val]

    predictions_train = [predict(p) for p in probs_train]
    predictions_val = [predict(p) for p in probs_val]

    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    if acc_final_tr < acc_train:
        best_param = point
        acc_final_tr = acc_train
        acc_final_val = acc_val
        best_seed = seed
        iteration = i

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.3f} | Acc validation: {:0.3f} "
        "".format(i + 1, cost(point, X_train, Y_train), acc_train, acc_val))


print("Final model: Cost: {:0.7f} | Acc train: {:0.3f} | Acc validation: {:0.3f} "
      "".format(cost(best_param, X_train, Y_train), acc_final_tr, acc_final_val))


file = open("results.csv", "a")
file.write("%s, %f, %f\n"%(dataset, acc_final_tr, acc_final_val))
file.close()