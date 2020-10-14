# qSLP: A Variational Algorithm for Quantum Neural Networks

This repository contains the code to reproduce the results in the paper [A Variational algorithm for Quantum Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-50433-5_45), 
published in the [International Conference on Computational Science 2020, Quantum Computing track](https://www.iccs-meeting.org/iccs2020/). 
# Description

In this work, we introduce a novel variational algorithm for quantum Single Layer Perceptron. 
Thanks to the universal approximation theorem, and given that the number of hidden neurons scales 
exponentially with the number of qubits, our framework opens to the possibility of approximating 
any function on quantum computers. Thus, the proposed approach produces a model with substantial 
descriptive power, and widens the horizon of potential applications already in the NISQ era, especially 
the ones related to Quantum Artificial Intelligence. In particular, we design a quantum circuit to perform 
linear combinations in superposition and discuss adaptations to classification and regression tasks. 
After this theoretical investigation, we also provide practical implementations using various simulation 
environments. Finally, we test the proposed algorithm on synthetic data exploiting both simulators and real 
quantum devices.

# Usage
The code is organised with one script per experiments in the paper. The **qml_** scripts use the framework [pennylane](https://pennylane.ai/)The qSLP algorithm allows reproducing the output of a Single Layer perceptron encoded in the amplitudes of a quantum state. In absence of a proper activation function a binary classification model on linearly separable data is estimated.

The script *qml_qSLP.py* contains the code for generating data and training a single classifier.

The script *qml_multiple_runs.py* generates many dataset from two bivariate distributions with different standard deviation and trains the classifier for each of them.

The script *qml-qiskit_real_device.py* executes the trained algorithm on real device, in the IBMQ quskit environment.

The script *qml_Utils.py* contains the import of the needed packages and all the custom routines for evaluation.

The script *qml_visualization.py* plots the data and the performances of the classifier.

The script *qml_collect_results_rl.py* collects the results.

Some of the functions for optimisation are taken from https://pennylane.ai/qml/demos/tutorial_variational_classifier.html


## Issues

For any issues or questions related to the code, open a new git issue or send a mail to antonio.macaluso2@unibo.it

