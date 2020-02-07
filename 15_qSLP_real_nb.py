from Utils_qml import *

X, y = datasets.make_blobs(n_samples=50, centers=[[0.2, 0.8],[0.7, 0.1]],
                           n_features=2, center_box=(0, 1),
                           cluster_std = 0.2) #, random_state = 56789)

# pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]

normalization = np.sqrt(np.sum(X_pad ** 2, -1))
X_norm = (X_pad.T / normalization).T

best_param = [[np.array([[[ 0.01763924,  0.01682827,  0.00978738],
                          [ 0.02240893,  0.01867558, -0.00977278]]]),
               np.array([[[ 1.13690040e-02, -2.00752973e+00, -1.03218852e-03],
                          [ 4.10598502e-03,  1.44043571e-03,  1.45427351e-02]]]),
               3.4269401874970473],
              0.0]

parameters = best_param[0]
bias = best_param[1]

devices = ['vigo', 'london', 'essex', 'burlington']
IBMQ_device = devices[1]


features = np.array([get_angles(x) for x in X_norm])
qiskit.IBMQ.load_account()

predictions_sim = []
predictions_qasm = []
predictions_real = []
i = 0
for f in features:
    #    f = features[1]
    device = qml.device("default.qubit", wires=5)
    pred_sim = test_qSLP_qml(f, best_param, dev= device)[0]
    predictions_sim.append(pred_sim)

    device = qml.device("qiskit.aer", wires=5, backend='qasm_simulator')
    pred_qasm = test_qSLP_qml(f, best_param, dev=device)[0]
    predictions_qasm.append(pred_qasm)

    device = qml.device('qiskit.ibmq', wires=5, backend='ibmq_'+IBMQ_device)
    pred_real = test_qSLP_qml(f, best_param, dev= device)[0]
    predictions_real.append(pred_real)

    data_test = pd.concat([pd.Series(predictions_sim),
                           pd.Series(predictions_qasm),
                           pd.Series(predictions_real)],
                           axis=1)

    data_test.to_csv('data.csv', index = False)
    i+=1
    print(i)

data_test = pd.concat([pd.Series(predictions_sim),
                       pd.Series(predictions_qasm),
                       pd.Series(predictions_real),
                       pd.Series(y)], axis=1)

data_test.columns = ['QML', 'QASM', 'Real_device', 'Y_true']
data_test.to_csv('results/data_'+ IBMQ_device + '.csv', index = False)

y_rl = np.where(data_test.Real_device>0, 1, 0)
y_qasm = np.where(data_test.QASM>0, 1, 0)
y_qml = np.where(data_test.QML>0, 1, 0)

print(IBMQ_device + ' MSE Real Device:', np.mean((y_rl-y)**2))
print( 'MSE QASM simulator: ', np.mean((y_qasm-y)**2))
print('MSE PennyLane simulator: ', np.mean((y_qml-y)**2))



data_test.to_csv('results/prediction_ibmq_'+IBMQ_device+'.csv', index = False)