# Classical packages
import jupyter
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

import math
from math import pi

from IPython.display import Image
from IPython.core.display import HTML

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
import pandas as pd


import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ, Aer
from qiskit import BasicAer, execute
from qiskit.tools.visualization import plot_state_city
from qiskit.providers.aer import StatevectorSimulator
from qiskit.tools.visualization import circuit_drawer
# from qiskit.transpiler import transpile

def create_dir (path):
    if not os.path.exists(path):
        print ('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        print ('The directory', path, ' already exists')


def save_dict(d, name = 'dict'):
    df = pd.DataFrame(list(d.items()))
    name = name + '_' + str(np.random.randint(10**6)) + '.csv'
    df.to_csv(name)


def normalize_custom(x, C =1):
    M = x[0] ** 2 + x[1] ** 2

    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed

def add_label( d, label = '0'):
    try:
        d[label]
        print( 'Label', label, 'exists')
    except:
        d[label] = 0
    return d

def exec_simulator(qc, n_shots = 8192):
    # QASM simulation
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots = n_shots)
    results = job.result()
    answer = results.get_counts(qc)
    return answer


def cos_classifier(x_train, x_new, y_train):
    # x_train = x1_train
    # x_new = x_test
    # y_train = y_class1
    c = ClassicalRegister(1)
    ancilla = QuantumRegister( 1 , 'yn')
    xt = QuantumRegister(1, 'x_train')
    xn = QuantumRegister(1, 'x_new')
    yt = QuantumRegister(1, 'y_train')
    qc = QuantumCircuit( xt, xn, ancilla, yt, c)
    qc.initialize(x_train, [ xt[0] ])
    qc.initialize(x_new, [ xn[0] ])
    qc.initialize(y_train, [ yt[0] ])
    qc.barrier()
    qc.h(ancilla)
    qc.cswap(ancilla, xn, xt)
    qc.h(ancilla)
    qc.barrier()
    qc.cx(yt, ancilla)
    qc.measure(ancilla, c)
    ####
    return qc

def plot_cls( dictionary, title = 'Test point classification' ):
    N = len(dictionary)
    fig, ax = plt.subplots()
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35         # the width of the bars
    prob_0 = [p['0']/(p['0'] + p['1']) for p in dictionary]
    prob_1 = [p['1']/(p['0'] + p['1']) for p in dictionary]
    label = [l['label'] for l in dictionary]
    pl1 = ax.bar(ind, prob_0, width, bottom=0)
    pl2 = ax.bar(ind + width, prob_1, width, bottom=0)
    ax.set_title( title )
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels( label )
    ax.legend((pl1[0], pl2[0]), ('P(y=0)', 'P(y=1)'))
    ax.autoscale_view()
    plt.show()



def pdf(url):
    return HTML('<embed src="%s" type="application/pdf" width="100%%" height="600px" />' % url)


# def qc_ensemble_full(D, x_test):
#     ''' Quantum Circuit for Ensemble '''
#     # dim(D) and label y1, y2, y3, y4 fixed
#     ''' Create Circuit '''
#     # Create a Classical Register with 1 bit.
#     c = ClassicalRegister(1)
#     # Create a Quantum Circuit
#     ancilla = QuantumRegister(2)
#     phi = QuantumRegister(9, 'phi')
#     psi = QuantumRegister(9, 'psi')
#     qc = QuantumCircuit(ancilla, psi, phi, c)
#
#     ancilla1 = ancilla[0]
#     ancilla2 = ancilla[1]
#
#     x1 = psi[0]
#     x2 = psi[1]
#     x3 = psi[2]
#     x4 = psi[3]
#     x_train = psi[4]
#
#     xt = psi[5]
#     yt = psi[6]
#
#     y_train = psi[7] # default class '1'
#     y_class0 = psi[8]
#
#     ### Initialization ###
#
#     qc.initialize(D[0], [x1])
#     # qc.barrier()
#     qc.initialize(D[1], [x2])
#     qc.initialize(D[2], [x3])
#     qc.initialize(D[3], [x4])
#     qc.initialize(x_test, [xt])
#     qc.x(y_train)
#     # Ancilla in superposition
#     qc.h(ancilla1)
#     qc.h(ancilla2)
#
#     qc.barrier()
#     qc.cswap(ancilla1, psi, phi )
#     qc.barrier()
#
#     ## +++++++++++++++++++ ##
#     # U1
#     qc.swap(x1, x3)
#
#     # U2
#     qc.swap(phi[1], phi[3])
#     ## +++++++++++++++++++ ##
#
#     qc.barrier()
#     qc.cswap(ancilla1, phi, psi)
#     qc.barrier()
#
#     qc.cswap(ancilla2, phi, psi)
#     qc.barrier()
#
#     ## +++++++++++++++++++ ##
#     # U3
#     qc.swap(x3, x_train)
#
#     # U4
#     qc.swap(phi[1], phi[4])
#     qc.swap(phi[7], phi[8])
#
#     ## +++++++++++++++++++ ##
#     qc.barrier()
#     qc.cswap(ancilla2, phi, psi)
#     qc.barrier()
#
#     # C
#     qc.h(yt)
#     qc.cswap(yt, xt, x_train)
#     qc.h(yt)
#     qc.cx(y_train, yt)
#     qc.measure(yt, c)
#     # print(qc)
#
#     ## Return circuit
#     return qc
#


def qc_ensemble_v2(D, x_test):
    x1_train = D[0]
    x2_train = D[1]
    x3_train = D[2]
    x4_train = D[3]
    ''' Quantum Circuit for Ensemble '''
    # dim(D) and label y1, y2, y3, y4 fixed
    ''' Create Circuit '''
    # Create a Classical Register with 1 bit.
    c = ClassicalRegister(1)
    # c2 = ClassicalRegister(1)
    # Create a Quantum Circuit
    ancilla = QuantumRegister(2)

    C1 = QuantumRegister(4, 'c1')
    C2 = QuantumRegister(4, 'c2')
    C3 = QuantumRegister(4, 'c3')
    C4 = QuantumRegister(4, 'c4')

    qc = QuantumCircuit(ancilla, C1, C2, C3, C4, c)

    ancilla1 = ancilla[0]
    ancilla2 = ancilla[1]

    ### Initialization ###
    qc.h(ancilla1)
    qc.h(ancilla2)

    x1 = C1[0]; qc.initialize(D[0], [x1]) # '1'
    x2 = C2[0]; qc.initialize(D[1], [x2]) # '0'
    x3 = C3[0]; qc.initialize(D[2], [x3]) # '1'
    x4 = C4[0]; qc.initialize(D[3], [x4]) # '0'

    x_test1 = C1[1]; qc.initialize(x_test, [x_test1])
    x_test2 = C2[1]; qc.initialize(x_test, [x_test2])
    x_test3 = C3[1]; qc.initialize(x_test, [x_test3])
    x_test4 = C4[1]; qc.initialize(x_test, [x_test4])

    y_test1 = C1[2]
    y_test2 = C2[2]
    y_test3 = C3[2]
    y_test4 = C4[2]

    y1 = C1[3]; qc.x( y1 )
    y2 = C2[3]
    y3 = C3[3]; qc.x( y3 )
    y4 = C4[3]

    qc.barrier()

    '''Computation'''
    # C1
    qc.h(y_test1)
    qc.cswap(y_test1, x_test1, x1)
    qc.h(y_test1)
    qc.cx(y1, y_test1)

    # C2
    qc.h(y_test2)
    qc.cswap(y_test2, x_test2, x2)
    qc.h(y_test2)
    qc.cx(y2, y_test2)

    # C1
    qc.h(y_test3)
    qc.cswap(y_test3, x_test3, x3)
    qc.h(y_test3)
    qc.cx(y3, y_test3)

    # C1
    qc.h(y_test4)
    qc.cswap(y_test4, x_test4, x4)
    qc.h(y_test4)
    qc.cx(y4, y_test4)
    qc.barrier()

    '''Collect results'''
    qc.cswap(ancilla2, y_test1, y_test2)
    qc.cswap(ancilla1, y_test1, y_test3)
    qc.cswap(ancilla1, y_test3, y_test4)
    qc.cswap(ancilla2, y_test1, y_test4)

    # qc.measure(ancilla1, c1)
    qc.measure(y_test1, c)# .c_if(c1, 1)
    # print(qc)
    return qc




def qc_ensemble_full(D, x_test):
    ''' Quantum Circuit for Ensemble '''
    # dim(D) and label y1, y2, y3, y4 fixed
    ''' Create Circuit '''
    # Create a Classical Register with 1 bit.
    c = ClassicalRegister(1)
    # Create a Quantum Circuit
    ancilla = QuantumRegister(2)
    phi = QuantumRegister(9, 'phi')
    psi = QuantumRegister(9, 'psi')
    qc = QuantumCircuit(ancilla, psi, phi, c)
    ancilla1 = ancilla[0]
    ancilla2 = ancilla[1]
    x1 = psi[0]
    x2 = psi[1]
    x3 = psi[2]
    x4 = psi[3]
    x_train = psi[4]
    xt = psi[5]
    yt = psi[6]
    y_train = psi[7]  # default class '1'
    y_class0 = psi[8]
    ### Initialization ###
    qc.initialize(D[0], [x1])
    # qc.barrier()
    qc.initialize(D[1], [x2])
    qc.initialize(D[2], [x3])
    qc.initialize(D[3], [x4])
    qc.initialize(x_test, [xt])
    qc.x(y_train)
    # Ancilla in superposition
    qc.h(ancilla1)
    qc.h(ancilla2)
    qc.barrier()
    for i in range(9):
        qc.cswap(ancilla1, psi[i], phi[i])
    qc.barrier()
    ## +++++++++++++++++++ ##
    # U1
    qc.swap(x1, x3)
    # U2
    qc.swap(phi[1], phi[3])
    ## +++++++++++++++++++ ##
    qc.barrier()
    for i in range(9):
        qc.cswap(ancilla1, psi[i], phi[i])
    qc.barrier()
    for i in range(9):
        qc.cswap(ancilla2, psi[i], phi[i])
    qc.barrier()
    ## +++++++++++++++++++ ##
    # U3
    qc.swap(x3, x_train)
    # U4
    qc.swap(phi[1], phi[4])
    qc.swap(phi[7], phi[8])
    ## +++++++++++++++++++ ##
    qc.barrier()
    for i in range(9):
        qc.cswap(ancilla2, psi[i], phi[i])
    qc.barrier()
    # C
    qc.h(yt)
    qc.cswap(yt, xt, x_train)
    qc.h(yt)
    qc.cx(y_train, yt)
    qc.measure(yt, c)
    # print(qc)
    # print(qc)
    ## Return circuit
    return qc
