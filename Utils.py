import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Qiskit
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ, Aer
from qiskit import BasicAer, execute
from qiskit.tools.visualization import plot_state_city
from qiskit.providers.aer import StatevectorSimulator
from qiskit.tools.visualization import circuit_drawer
from qiskit.circuit import Parameter
from qiskit.circuit import Parameter

# Pennylane
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn import datasets

def get_angles(x):
    """
    Determine the angles of rotation for a specific set of features
    :param x: (float) input vector
    :return: (float) vector of parameters for rotations
    """

    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def statepreparation(a):
    '''
    State preparation routine to encode data in amplitude encoding
    :param a: (float) angles
    :return: circuit to encode a vector in amplitude encoding
    '''
    qml.RY(a[0], wires=1)

    qml.CNOT(wires=[1, 2])
    qml.RY(a[1], wires=2)
    qml.CNOT(wires=[1, 2])
    qml.RY(a[2], wires=2)

    qml.PauliX(wires=1)
    qml.CNOT(wires=[1, 2])
    qml.RY(a[3], wires=2)
    qml.CNOT(wires=[1, 2])
    qml.RY(a[4], wires=2)
    qml.PauliX(wires=1)

def qiskit_state_prep(qc, qr, a):
    qc.ry(a[0], qr[1])
    #qml.RY(a[0], wires=1)

    # qml.CNOT(wires=[1, 2])
    qc.cx(qr[0], qr[1])

    # qml.RY(a[1], wires=2)
    qc.ry(a[1], qr[1])

    # qml.CNOT(wires=[1, 2])
    qc.cx(qr[0], qr[1])

    # qml.RY(a[2], wires=2)
    qc.ry(a[2], qr[1])

    # qml.PauliX(wires=1)
    qc.x(qr[0])

    # qml.CNOT(wires=[1, 2])
    qc.cx(qr[0], qr[1])

    # qml.RY(a[3], wires=2)
    qc.ry(a[3], qr[1])

    #qml.CNOT(wires=[1, 2])
    qc.cx(qr[0], qr[1])

    # qml.RY(a[4], wires=2)
    qc.ry(a[4], qr[1])

    # qml.PauliX(wires=1)
    qc.x(qr[0])


def square_loss(labels, predictions):
    """
    Compute the square loss between the true value (labels) and the predictions
    :param labels: (integer) true value for the target variable
    :param predictions: (float) prediction of the classification model
    :return: (float) square loss between true value and prediction
    """
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss



def accuracy(labels, predictions):
    """
     Compute the accuracy of a given prediction based on the true value (labels)
    :param labels: (integer) true value for the target variable
    :param predictions: (float) prediction of the classification model
    :return: (float) accuracy of the prediction
    """
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss




def layer(W, wires = None):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=wires[0])
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def normalize_custom(x, C =1):
    M = x[0] ** 2 + x[1] ** 2

    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed


def test_qSLP_qiskit(x, param_circuit, device = 'qasm_simulator'):
    theta_11 = param_circuit[0][0][0]  # array([ 0.01762722, -0.05147767,  0.00978738])
    theta_12 = param_circuit[0][0][1]  # array([ 0.02240893,  0.01867558, -0.00977278])
    theta_21 = param_circuit[1][0][0]  # array([ 5.60373788e-03, -1.11406652e+00, -1.03218852e-03])
    theta_22 = param_circuit[1][0][1]  # array([0.00410599, 0.00144044, 0.01454274])
    beta = param_circuit[2]

        # x = X_norm[i]
    ''' Create Circuit '''
    # Create a Classical Register with 1 bit.
    c = ClassicalRegister(1)
    # Create a Quantum Circuit
    control = QuantumRegister(1)
    data = QuantumRegister(2)
    temp = QuantumRegister(2)

    qc = QuantumCircuit(control, data, temp, c)

    qiskit_state_prep(qc, data, x)

    ### Initialization ###
    qc.ry(beta, control)

    # qc.initialize(x, [data])

    qc.h(temp)

    qc.barrier()
    '''Computation'''
    qc.cswap(control, data[0], temp[0])
    qc.cswap(control, data[1], temp[1])

    # First layer
    #theta_11 = param_circuit[0][0][0] #array([ 0.01762722, -0.05147767,  0.00978738])
    qc.rz(theta_11[0], data[0] )
    qc.ry(theta_11[1], data[0] )
    qc.rz(theta_11[2], data[0] )

    # theta_12 = param_circuit[0][0][1] #array([ 0.02240893,  0.01867558, -0.00977278])
    qc.rz(theta_12[0], data[1] )
    qc.ry(theta_12[1], data[1] )
    qc.rz(theta_12[2], data[1] )

    qc.cx(data[0], data[1])

    # Second layer
    # theta_21 = param_circuit[1][0][0] #array([ 5.60373788e-03, -1.11406652e+00, -1.03218852e-03])
    qc.rz(theta_21[0], temp[0] )
    qc.ry(theta_21[1], temp[0] )
    qc.rz(theta_21[2], temp[0] )

    # theta_22 = param_circuit[1][0][1] # array([0.00410599, 0.00144044, 0.01454274])
    qc.rz(theta_22[0], temp[1] )
    qc.ry(theta_22[1], temp[1] )
    qc.rz(theta_22[2], temp[1] )

    qc.cx(temp[0], temp[1])

    qc.cswap(control, data[0], temp[0])
    qc.cswap(control, data[1], temp[1])

    #print(qc)

    # qc.measure(ancilla1, c1)
    qc.measure(data[0], c)  # .c_if(c1, 1)
    # print(qc)
    backend = BasicAer.get_backend(device)
    job = execute(qc, backend, shots = 8192)
    results = job.result()
    answer = results.get_counts(qc)
    # if answer['0'] > answer['1']:
    #     y_pred = 1
    # else:
    #     y_pred = 0
    y_pred = np.sum(answer['1']*(-1)+answer['0']*(1))/(answer['0']+answer['1'])
    return [y_pred, qc]




def multivariateGrid(col_x, col_y, col_k, df, col_color=None,
                     scatter_alpha=0.5):
    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt


    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends = []
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        # if col_color:
        #     colors_data = np.unique(df[col_color])
        # else:
        #     colors_data = ["or_blue", "or_peru"]

        if col_color:
            color = df_group[col_color].tolist()[0]
        g.plot_joint(
            colored_scatter(df_group[col_x], df_group[col_y], color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,
            vertical=True
        )
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey'
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True
    )
    plt.tight_layout()
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20, rotation=0)
    plt.legend(legends, fontsize=18, loc='lower left')
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    colors_data = np.unique(df[col_color])
    plt.savefig('results/Data_{}_{}.png'.format(
        colors_data[0][:2], colors_data[1][:2]), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()




