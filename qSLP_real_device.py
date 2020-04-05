from Utils import *

X, y = datasets.make_blobs(n_samples=50, centers=[[0.2, 0.8],[0.7, 0.1]],
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


features = np.array([get_angles(x) for x in X_norm])
qiskit.IBMQ.load_account()

predictions_qml_sim = []
predictions_qml_real = []

for f in features:
#    f = features[1]
    device = qml.device("default.qubit", wires=5)
    pred_sim = test_qSLP_qml(f, best_param, dev= device)[0]
    predictions_qml_sim.append(pred_sim)


    device = qml.device('qiskit.ibmq', wires=5, backend='ibmq_london')
    pred_real = test_qSLP_qml(f, best_param, dev= device)[0]
    predictions_qml_real.append(pred_real)
    # row = f.tolist()
    # row.append(pred_sim)
    # row.append(pred_real)
    # row = pd.Series(row)
    data_test = pd.concat([pd.Series(predictions_qml_sim),
                           pd.Series(predictions_qml_real)],
                           axis=1)
    data_test.to_csv('data_test.csv')



data_test = pd.concat([pd.Series(predictions_qml_sim),
                       pd.Series(predictions_qml_real),pd.Series(y)], axis=1)

data_test.to_csv('data_test.csv')


predictions_qasm = []
predictions_qml = []

for f in features:
#    f = features[1]
    device = qml.device("qiskit.aer", wires=5, backend='qasm_simulator')
    pred_qasm = test_qSLP_qml(f, best_param, dev= device)[0]
    predictions_qasm.append(pred_qasm)

    device = qml.device("default.qubit", wires=5)
    pred_qml = test_qSLP_qml(f, best_param, dev= device)[0]
    predictions_qml.append(pred_qml)

    data_test_qasm = pd.concat([pd.Series(predictions_qasm),
                                pd.Series(predictions_qml)],
                                axis=1)
    data_test_qasm.to_csv('data_test_qasm.csv', index=False)



data_test_qasm = pd.concat([pd.Series(predictions_qml_sim),
                            pd.Series(predictions_qml_real),pd.Series(y)],
                            axis=1)

data_test_qasm.to_csv('data_test_qasm.csv', index = False)





# provider = qiskit.IBMQ.get_provider(group='open')
# ibmq_ourense = provider.get_backend('ibmq_ourense')
#
# backend = IBMQ.backends(operational=True, simulator=False)[2]
# pred = test_qSLP_qml(f, best_param)[0]