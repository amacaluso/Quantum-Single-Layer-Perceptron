import os.path, sys
dir = os.path.join('qiskit implementation')
sys.path.insert(0, dir)

from Utils import *


def load_iris(fraction=1, plot=True, type=1):

    if type==1:
        df = np.loadtxt("qiskit implementation/data/iris_classes1and2_scaled.txt")
    else:
        df = np.loadtxt("qiskit implementation/data/iris_classes2and3_scaled.txt")

    df = pd.DataFrame(df)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Y']

    df = df.sample(frac=fraction)

    X = df.iloc[:, 0:2].values
    Y = (df.iloc[:, -1] + 1) / 2
    Y = Y.to_numpy()

    if plot:
        sns.set_theme(style="ticks")
        sns.pairplot(df.iloc[:, [0,1,4]], hue="Y")
        plt.show()
    return X,Y


def load_bivariate_gaussian(n_train=20, plot=True):
    X, Y = datasets.make_blobs(n_samples=n_train, centers=[[0.3, 0.8], [0.7, 0.3]], # centers=[[0.3, 0.8], [0.7, 0.3]]
                               n_features=2, center_box=(0, 1), cluster_std=0.1,
                               random_state=543)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['Y'] = Y
    if plot:
        sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y")
        plt.show()
    return X,Y



def load_parity(plot=True):
    df = np.loadtxt("qiskit implementation/data/parity.txt")
    df = pd.DataFrame(df)
    df.columns = ['X' + str(i) for i in range(df.shape[1] - 1)] + ['Y']
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    #Y = Y * 2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}

    if plot:
        sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y")
        plt.show()
    return X,Y


def load_moon(fraction=1, plot=True, type=1):

    df = np.loadtxt("qiskit implementation/data/moons.txt")

    df = pd.DataFrame(df)
    df.columns = ['X' + str(i) for i in range(df.shape[1] - 1)] + ['Y']

    df = df.sample(frac=fraction)

    X = df.iloc[:, 0:2].values
    Y = df.iloc[:, 2].values

    if plot:
        sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y")
        plt.show()
    return X,Y

###################################################################
###################################################################


def load_blood_transfusion(plot=True):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data')
    df.columns = ['R', 'F', 'M', 'T', 'Y']

    Y = df.Y
    X = df.drop(labels='Y', axis=1).values #returns a numpy array
    min_max_scaler = preprocessing.StandardScaler()
    X_scaled = min_max_scaler.fit_transform(X)


    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    x_new = pca.fit_transform(X)
    print('Explained variance of the 2 firts components:',
          pca.explained_variance_ratio_)
    df = pd.DataFrame(x_new, columns=['x1', 'x2'])
    df['Y'] = Y

    if plot:
        sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y")
        plt.show()
    return df


def load_banknote_authentication(plot=True, fraction=.1):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')
    df.columns = ['X'+ str(i) for i in range(df.shape[1]-1)] + ['Y']

    g = df.groupby('Y')
    df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    df = df.sample(frac=fraction)

    Y = df.Y.to_numpy()
    X = df.drop(labels='Y', axis=1).values #returns a numpy array
    min_max_scaler = preprocessing.StandardScaler()
    X_scaled = min_max_scaler.fit_transform(X)


    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    x_new = pca.fit_transform(X)
    x_new_scaled = min_max_scaler.fit_transform(x_new)
    print('Explained variance of the 2 firts components:',
          pca.explained_variance_ratio_)
    df = pd.DataFrame(x_new_scaled, columns=['x1', 'x2'])
    df['Y'] = Y

    if plot:
        sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y")
        plt.show()
    return x_new_scaled, Y


def load_haberman(plot=True):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data')
    df.columns = ['age', 'year', 'num_pos', 'Y']
    df.Y.value_counts()

    Y = df.Y - 1
    X = df.drop(labels='Y', axis=1).values #returns a numpy array
    min_max_scaler = preprocessing.StandardScaler()
    X_scaled = min_max_scaler.fit_transform(X)


    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    x_new = pca.fit_transform(X)
    print('Explained variance of the 2 firts components:',
          pca.explained_variance_ratio_)
    df = pd.DataFrame(x_new, columns=['x1', 'x2'])
    df['Y'] = Y

    if plot:
        sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y")
        plt.show()
    return df


# def load_iris(plot=True, class_to_drop = 'Iris-setosa', fraction = .2):
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
#     df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Y']
#     df = df[df.Y != class_to_drop]
#     df = df.sample(frac=fraction)
#
#     Y = pd.factorize(df.Y)[0]
#     X = df.drop(labels=['Y', 'petal_length', 'petal_width'] , axis=1).values #returns a numpy array
#
#     # PCA
#     min_max_scaler = preprocessing.StandardScaler()
#     X_scaled = min_max_scaler.fit_transform(X)
#     # pca = PCA(n_components=2)
#     # pca.fit(X_scaled)
#     # x_new = pca.fit_transform(X)
#     # print('Explained variance of the 2 firts components:',
#     #       pca.explained_variance_ratio_)
#     # df = pd.DataFrame(x_new, columns=['x1', 'x2'])
#     # df['Y'] = Y
#     # df = df[df.Y != class_to_drop]
#
#     if plot:
#         sns.set_theme(style="ticks")
#         sns.pairplot(df, hue="Y")
#         plt.show()
#     return X_scaled,Y
