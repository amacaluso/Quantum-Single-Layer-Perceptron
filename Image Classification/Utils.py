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
import random

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings("ignore")



def predict(probas):
    return (probas >= 0.5) * 1


def binary_crossentropy(labels, predictions):
    '''
    Compare a set of predictions and the real values to compute the Binary Crossentropy for a binary target variable
    :param labels: true values for a binary target variable.
    :param predictions: predicted probabilities for a binary target variable
    :return: the value of the binary cross entropy. The lower the value is, the better are the predictions.
    '''
    loss = 0
    for l, p in zip(labels, predictions):
        # print(l,p)
        loss = loss - l * np.log(np.max([p, 1e-8]))

    loss = loss / len(labels)
    return loss



def square_loss(labels, predictions):
    '''
    Compare a set of predictions and the real values to compute the Mean Squared Error for a binary target variable
    :param labels: true values for a binary target variable.
    :param predictions: predicted probabilities for a binary target variable
    :return: the value of the binary cross entropy. The lower the value is, the better are the predictions.
    '''
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    '''
    Compare a set of predictions and the real values to compute the Accuracy for a binary target variable
    :param labels: true values for a binary target variable.
    :param predictions: predicted values for a binary target variable
    :return: the value of the binary cross entropy. The lower the value is, the better are the predictions.
    '''
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss



def get_angles(x):
    '''
    Given a real vector computes the rotation angles for state preparation
    :param labels: positive 4 dimensional real vector
    :return: rotation angles
    '''
    beta0 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def plot_images(images, labels, num_row, num_col):
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()


def scatterplot_matrix(data):
    # sns.set_theme(style="ticks")
    sns.pairplot(data, hue="Y")
    plt.show()



def image_reshape(x, ncol, pca, lower, upper):
    x = x.reshape(-1, ncol)
    x = pca.transform(x)
    x = (x - lower) / (upper - lower)

    padding = 0.3 * np.ones((len(x), 1))
    x_pad = np.c_[np.c_[x, padding], np.zeros((len(x), 1))]

    normalization = np.sqrt(np.sum(x_pad ** 2, -1))
    x_norm = (x_pad.T / normalization).T

    features = np.nan_to_num((np.array([get_angles(x) for x in x_norm])))
    return features[0].tolist()

