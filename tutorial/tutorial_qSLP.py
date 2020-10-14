import os.path, sys
dir = os.path.join('tutorial')
sys.path.insert(0, dir)

from Utils import *
from import_data import *

# X, Y = load_iris(fraction=.5)
X, Y = load_bivariate_gaussian(n_train=100)
# X,Y = load_parity(plot=True)
# X,Y = load_moon(fraction=.4, plot=True)


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



# normalize each input
normalization = np.sqrt(np.sum(X ** 2, -1))
X_norm = (X.T / normalization).T
print("First X sample (normalized):", X_norm[0])


def get_Sx(ang=None, x=None, pad=False):
    backend = Aer.get_backend('unitary_simulator')

    if pad==True:
        q = QuantumRegister(2)
        circuit = QuantumCircuit(q)
        circuit = state_preparation(ang, circuit, [0, 1])
    elif pad==False:
        x = x.astype(complex)
        q = QuantumRegister(1)
        circuit = QuantumCircuit(q)
        circuit.initialize(x, [q])

    job = execute(circuit, backend)
    result = job.result()
    U = result.get_unitary(circuit)
    S = Operator(U)
    return S



def linear_operator(param, pad=False):
    backend = Aer.get_backend('unitary_simulator')
    '''pad variable influences the size of params vector'''
    if pad==True:
        data = QuantumRegister(2)
        qc = QuantumCircuit(data_reg)
        qc.u3(param[0], param[1], param[2], data[0])
        qc.u3(param[3], param[4], param[5], data[1])
        qc.cx(data[0], data[1])
    elif pad==False:
        data = QuantumRegister(1)
        qc = QuantumCircuit(data)
        qc.u3(param[0], param[1], param[2], data)

    job = execute(qc, backend)
    result = job.result()

    U = result.get_unitary(qc)
    G = Operator(U)
    return G

def sigma(pad=False):
    backend = Aer.get_backend('unitary_simulator')
    if pad==True:
        data = QuantumRegister(2)
        qc = QuantumCircuit(data)
        qc.id(data)
    if pad==False:
        data = QuantumRegister(1)
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

def create_circuit(parameters=None, x=None, pad=False):
    beta = parameters[0]
    theta1 = parameters[1:4]
    theta2 = parameters[4:7]

    control = QuantumRegister(1, 'control')
    data = QuantumRegister(1, 'x')
    temp = QuantumRegister(1, 'temp')
    c = ClassicalRegister(1)
    qc = QuantumCircuit(control, data, temp, c)

    S=get_Sx(x=x)
    qc.unitary(S, data, label='$S_{x}$')

    R=R_gate(beta)
    qc.unitary(R, control, label='$R_{Y}(β)$')

    qc.barrier()
    qc.cswap(control, data, temp)

    G1 = linear_operator(theta1)
    qc.unitary(G1, data, label='$G(θ_{1})$')

    G2 = linear_operator(theta2)
    qc.unitary(G2, temp, label='$G(θ_{2})$')

    qc.cswap(control, data, temp)
    qc.barrier()

    sig = sigma()
    qc.unitary(sig, data, label='$Σ$')

    qc.barrier()
    qc.measure(data[0], c)
    return qc

c=create_circuit(parameters=range(7), x=np.array([0,1]))
c.draw(output='mpl')
plt.show()

def execute_circuit(parameters, x=None, shots=1000, print=False):
    backend = BasicAer.get_backend('qasm_simulator')

    circuit=create_circuit(parameters, x)
    if print:
        circuit.draw(output='mpl')
        plt.show()
    result = execute(circuit, backend, shots=shots).result()

    counts = result.get_counts(circuit)
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
var = (0.001*np.random.randn(7))
current_params = var

from qiskit.aqua.components.optimizers import AQGD
optimizer_step = AQGD(maxiter=1, eta=2.0, disp=False)
execute_circuit(current_params, x=X[2], print=True)


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
    point, value, nfev = optimizer_step.optimize(len(current_params), obj_function,
                                                 initial_point=current_params)
    current_params = point

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