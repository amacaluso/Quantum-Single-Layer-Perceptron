from Utils_qml import *
from sklearn import datasets


dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def circuit(weights, angles=None):
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
    weights = var[0]
    bias = var[1]
    return circuit(weights, angles=angles) + bias


def cost(weights, features, labels):
    predictions = [variational_classifier(weights, angles=f) for f in features]
    return square_loss(labels, predictions)

std_dev = np.arange(0.05, 0.5, 0.02)
n = len(std_dev)
seeds = np.random.randint(1, 10**5, n)
tr_accuracy_vector =[]
vl_accuracy_vector =[]
cost_vector = []


for i in range(n):
    print(i)
    X, y = datasets.make_blobs(n_samples = 100, centers = [[0.2, 0.8],[0.7, 0.1]] ,
                               n_features=2, center_box=(0, 1),
                               cluster_std = std_dev[i], random_state = seeds[i])
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.show()
    Y = np.where(y == 0, -1, 1)

    padding = 0.3 * np.ones((len(X), 1))
    X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
    print("First X sample (padded)    :", X_pad[0])

    # normalize each input
    normalization = np.sqrt(np.sum(X_pad ** 2, -1))
    X_norm = (X_pad.T / normalization).T
    print("First X sample (normalized):", X_norm[0])

    # angles for state preparation are new features
    features = np.array([get_angles(x) for x in X_norm])

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

    opt = NesterovMomentumOptimizer(0.01)
    batch_size = 10

    acc_final_tr = 0
    acc_final_val = 0
    num_qubits = 2
    num_layers = 1

    var_init = ((0.01 * np.random.randn(num_layers, num_qubits, 3),
                 0.01 * np.random.randn(num_layers, num_qubits, 3),
                 2*np.pi*np.random.random_sample()),
                0.0)
    var = var_init

    best_param = var_init
    for it in range(50):

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
            cost_final = cost(var, features, Y)
            iteration = it

        print(
            "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
            "".format(it + 1, cost(var, features, Y), acc_train, acc_val)
        )
    tr_accuracy_vector.append(acc_final_tr)
    vl_accuracy_vector.append(acc_final_val)
    cost_vector.append(cost_final)

    var
    var_init
    best_param



std_dev
tr_accuracy_vector
vl_accuracy_vector
cost_vector

plt.plot(std_dev,vl_accuracy_vector, 'g^')
plt.plot(std_dev, tr_accuracy_vector, 'bs')
plt.plot(std_dev, cost_vector)
plt.legend()
plt.show()
Y = np.where(y == 0, -1, 1)

# ##############################################################################
    # # We can plot the continuous output of the variational classifier for the
    # # first two dimensions of the Iris data set.
    #
    # plt.figure()
    # cm = plt.cm.RdBu
    #
    # # make data for decision regions
    # xx, yy = np.meshgrid(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))
    # X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]
    #
    # # preprocess grid points like data inputs above
    # padding = 0.3 * np.ones((len(X_grid), 1))
    # X_grid = np.c_[np.c_[X_grid, padding], np.zeros((len(X_grid), 1))]  # pad each input
    # normalization = np.sqrt(np.sum(X_grid ** 2, -1))
    # X_grid = (X_grid.T / normalization).T  # normalize each input
    # features_grid = np.array(
    #     [get_angles(x) for x in X_grid]
    # )  # angles for state preparation are new features
    # predictions_grid = [variational_classifier(best_param, angles=f) for f in features_grid]
    # Z = np.reshape(predictions_grid, xx.shape)
    #
    # # plot decision regions
    # cnt = plt.contourf(xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=0.8, extend="both")
    # plt.contour(xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,))
    # plt.colorbar(cnt, ticks=[-1, 0, 1])
    #
    #
    # # plot data
    # plt.scatter(
    #     X_train[:, 0][Y_train == 1],
    #     X_train[:, 1][Y_train == 1],
    #     c="b",
    #     marker="o",
    #     edgecolors="k",
    #     label="class 1 train",
    # )
    # plt.scatter(
    #     X_val[:, 0][Y_val == 1],
    #     X_val[:, 1][Y_val == 1],
    #     c="b",
    #     marker="^",
    #     edgecolors="k",
    #     label="class 1 validation",
    # )
    # plt.scatter(
    #     X_train[:, 0][Y_train == -1],
    #     X_train[:, 1][Y_train == -1],
    #     c="r",
    #     marker="o",
    #     edgecolors="k",
    #     label="class -1 train",
    # )
    # plt.scatter(
    #     X_val[:, 0][Y_val == -1],
    #     X_val[:, 1][Y_val == -1],
    #     c="r",
    #     marker="^",
    #     edgecolors="k",
    #     label="class -1 validation",
    # )
    #
    # plt.legend()
    # plt.show()