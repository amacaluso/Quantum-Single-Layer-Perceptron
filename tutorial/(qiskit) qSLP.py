import os.path, sys
dir = os.path.join('tutorial')
sys.path.insert(0, dir)

from qml_Utils import *
from modeling import *
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



def get_Sx(ang, x=None, pad=True):
    backend = Aer.get_backend('unitary_simulator')

    q = QuantumRegister(2)
    circuit = QuantumCircuit(q)
    circuit = state_preparation(ang, circuit, [0, 1])

    job = execute(circuit, backend)
    result = job.result()

    U = result.get_unitary(circuit)
    S = Operator(U)
    return S


x = X_norm[0]
ang = get_angles(x)
q = QuantumRegister(2)
circuit = QuantumCircuit(q)
circuit = state_preparation(ang, circuit, [0, 1])
circuit.draw(output='mpl')
plt.show()


def linear_operator(param):
    backend = Aer.get_backend('unitary_simulator')

    data_reg = QuantumRegister(2)
    qc = QuantumCircuit(data_reg)
    qc.u3(param[0], param[1], param[2], data_reg[0])
    qc.u3(param[3], param[4], param[5], data_reg[1])
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

    backend = BasicAer.get_backend('qasm_simulator')

    param0 = parameters[0]
    param1 = parameters[1:7]
    param2 = parameters[7:13]

    control = QuantumRegister(1, 'control')
    data = QuantumRegister(2, 'x')
    temp = QuantumRegister(2, 'temp')
    c = ClassicalRegister(1)
    qc = QuantumCircuit(control, data, temp, c)

    ang = np.nan_to_num(get_angles(x))
    S = get_Sx(ang)
    qc.unitary(S, data, label='$S_{x}$')

    R = R_gate(param0)
    qc.unitary(R, control, label='$R_{Y}(β)$')

    qc.barrier()
    qc.cswap(control, data[0], temp[0])
    qc.cswap(control, data[1], temp[1])

    G1 = linear_operator(param1)
    qc.unitary(G1, data, label='$G(θ_{1})$')


    G2 = linear_operator(param2)
    qc.unitary(G2, temp, label='$G(θ_{2})$')

    qc.cswap(control, data[1], temp[1])
    qc.cswap(control, data[0], temp[0])

    sig = sigma()
    qc.unitary(sig, data, label='$Σ$')

    qc.barrier()
    qc.measure(data[0], c)
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

# var_init = (0.01*np.random.randn(1), 0.01*np.random.randn(7))
seed = 974# np.random.randint(0,10**3,1)[0]
print(seed)
np.random.seed(seed)
var = (0.1*np.random.randn(13))
params_init = var
### iris --> 359
### gaussian --> 527
num_data = len(Y)
num_train = int(0.75 * num_data)
index = np.random.permutation(range(num_data))
X_train = X[index[:num_train]]
Y_train = Y[index[:num_train]]
X_val = X[index[num_train:]]
Y_val = Y[index[num_train:]]
#

from qiskit.aqua.components.optimizers import AQGD
optimizer = AQGD(maxiter=20, eta=2.0, disp=True)
execute_circuit(params_init, x=X[2], print=True)



obj_function = lambda params: cost(params, X_train, Y_train)
# init_param = np.array(params)

point, value, nfev = optimizer.optimize(len(params_init), obj_function, initial_point=params_init)

best_params = point

probs_train = [execute_circuit(best_params, x) for x in X_train]
probs_val = [execute_circuit(best_params, x) for x in X_val]


predictions_train = [predict(p) for p in probs_train]
predictions_val = [predict(p) for p in probs_val]


acc_train = accuracy(Y_train, predictions_train)
print('train accuracy',acc_train)

acc_val = accuracy(Y_val, predictions_val)
print('val accuracy',acc_val)
