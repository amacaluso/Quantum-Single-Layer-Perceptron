# qSLP: A Variational Algorithm for Quantum Neural Networks

This repository contains the code to reproduce the results in the paper A Variational algorithm for Quantum Neural Networks, accepted in the INTERNATIONAL CONFERENCE ON COMPUTATIONAL SCIENCE 2020, Quantum Computing track. 

# Description
The code is organised with one script per experiments in the paper. The qSLP algorithm allows reproducing the output of a Single Layer perceptron encoded in the amplitudes of a quantum state. In absence of a proper activation function a binary classification model on linearly separable data is estimated.

The script *qSLP.py* contains the code for generating data and training a single classifier.

The script *qSLP_multiple_runs.py* generates many dataset from two bivariate distributions with different standard deviation and trains the classifier for each of them.

The script *qSLP_real_device.py* executes the trained algorithm on real device, in the IBMQ quskit environment.

The script *Utils.py* contains the import of the needed packages and all the custom routines for evaluation.

The script *Viz_data_&_performance.py* plots the data and the performances of the classifier.

The script *tab_results_real_device.py* collects the results.

Some of the funcions for optimisation are taken from https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
