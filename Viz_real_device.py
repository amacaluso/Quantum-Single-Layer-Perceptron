from Utils import *

X, y = datasets.make_blobs(n_samples=50, centers=[[0.2, 0.8],[0.7, 0.1]],
                           n_features=2, center_box=(0, 1),
                           cluster_std = 0.2, random_state = 5432)

data_vigo = pd.read_csv('results/data_vigo.csv')
data_essex = pd.read_csv('results/data_essex.csv')
data_london = pd.read_csv('results/data_london.csv')
data_burlington = pd.read_csv('results/data_burlington.csv')


y_london = np.where(data_london.Real_device>0, 1, 0)
y_vigo = np.where(data_vigo.Real_device>0, 1, 0)
y_essex = np.where(data_essex.Real_device>0, 1, 0)
y_burlington = np.where(data_burlington.Real_device>0, 1, 0)

y_qasm = np.where(data_vigo.QASM>0, 1, 0)
y_qml = np.where(data_vigo.QML>0, 1, 0)


#print('London:', np.mean((y_london-y)**2))
print('Vigo:', np.mean((y_vigo-y)**2))
print('Essex:', np.mean((y_essex-y)**2))
print('Burlington:', np.mean((y_burlington-y)**2))

print( 'MSE QASM simulator: ', np.mean((y_qasm-y)**2))
print('MSE PennyLane simulator: ', np.mean((y_qml-y)**2))

