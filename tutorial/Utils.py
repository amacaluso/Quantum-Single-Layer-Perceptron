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


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from qiskit import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute, IBMQ, Aer
from qiskit.circuit import Gate
from qiskit.quantum_info.operators import Operator
from qiskit.aqua.components.optimizers import ADAM, CG, AQGD



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
    # plt.savefig('results/Data_{}_{}.png'.format(
    #     colors_data[0][:2], colors_data[1][:2]), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# def state_preparation(x):
#     backend = Aer.get_backend('unitary_simulator')
#
#     x = normalize_custom(x)
#
#     qreg = QuantumRegister(1)
#     qc = QuantumCircuit(qreg)
#     # Run the quantum circuit on a unitary simulator backend
#     qc.initialize(x, [qreg])
#     job = execute(qc, backend)
#     result = job.result()
#
#     U = result.get_unitary(qc)
#     S = Operator(U)
#     return S


def predict(probas):
    return (probas >= 0.5) * 1


def binary_crossentropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        # print(l,p)
        loss = loss - l * np.log(np.max([p, 1e-8]))

    loss = loss / len(labels)
    return loss


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss



def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])



def state_preparation(a, circuit, target):
    a = 2 * a
    circuit.ry(a[0], target[0])

    circuit.cx(target[0], target[1])
    circuit.ry(a[1], target[1])
    circuit.cx(target[0], target[1])
    circuit.ry(a[2], target[1])

    circuit.x(target[0])
    circuit.cx(target[0], target[1])
    circuit.ry(a[3], target[1])
    circuit.cx(target[0], target[1])
    circuit.ry(a[4], target[1])
    circuit.x(target[0])

    return circuit



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
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    colors_data = np.unique(df[col_color])
    plt.savefig('Data_{}_{}.png'.format(
        colors_data[0][:2], colors_data[1][:2]), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
