from Utils import *
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=10, centers=[[0.2, 0.8],[0.7, 0.1]],
                           n_features=2, center_box=(0, 1),
                           cluster_std = 0.2, random_state = 5432)

# pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]

normalization = np.sqrt(np.sum(X_pad ** 2, -1))
X_norm = (X_pad.T / normalization).T


best_param = [[np.array([[[ 0.01762722, -0.05147767,  0.00978738],
                          [ 0.02240893,  0.01867558, -0.00977278]]]),
               np.array([[[ 5.60373788e-03, -1.11406652e+00, -1.03218852e-03],
                          [ 4.10598502e-03,  1.44043571e-03,  1.45427351e-02]]]),
               3.4785004378680453],
              -0.7936398118318136]

parameters = best_param[0]
bias = best_param[1]


''' State Preparation'''
# Training Set
features = np.array([get_angles(x) for x in X_norm])
predictions_qiskit = []
for f in features:
    pred = test_qSLP_qiskit(f, param_circuit=parameters)[0] + bias
    predictions_qiskit.append(pred)

print(test_qSLP_qiskit(f, param_circuit=parameters)[1])

predictions_qiskit = np.array(predictions_qiskit)
pred_labels  = np.where(predictions_qiskit>0, 1, 0)
np.mean([(y_true-p)**2 for y_true, p in zip(y, pred_labels)])

# # execution
# if device == 'qasm_simulator':
#    backend = BasicAer.get_backend(device)
# else:
#     backend = device
# job = execute(qc, backend, shots = n_shots)
# results = job.result()
# answer = results.get_counts(qc)
# print(answer)
# print(answer['0']/sum(answer.values()))





features = np.array([get_angles(x) for x in X_norm])
predictions_qml = []

for f in features:
    pred = test_qSLP_qml(f, best_param)[0]
    predictions_qml.append(pred)

predictions_qml = np.array(predictions_qml)
pred_labels  = np.where(predictions_qml > 0, 1, 0)
np.mean([(y_true-p)**2 for y_true, p in zip(y, pred_labels)])


plt.scatter(predictions_qiskit, predictions_qml, c = y)
plt.show()


import pennylane as qml
import qiskit
from qiskit.providers.aer.noise.device import basic_device_noise_model

qiskit.IBMQ.load_account()
provider = qiskit.IBMQ.get_provider(group='open')
ibmq_ourense = provider.get_backend('ibmq_ourense')

backend = IBMQ.backends(operational=True, simulator=False)[2]

dev = qml.device('qiskit.ibmq', wires=5, backend = 'ibmq_ourense')
pred = test_qSLP_qml(f, best_param)[0]