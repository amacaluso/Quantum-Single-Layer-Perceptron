import numpy as np
import matplotlib.pylab as plt


def step(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y_step = step(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)

plt.plot(x, y_step, label=r'Step: $0_{\chi_{(-\infty,0]}}+ 1_{\chi_{[0, \infty)}}$', color='green', lw=1, linestyle=None)
plt.plot(x, y_sigmoid, label=r'Sigmoid: $\frac{1}{1+e^{-x}}$', color='b', lw=1, ls='--')
plt.ylim(-0.1, 1.1)
plt.grid(alpha = 0.2)
plt.ylabel(r'$f(x)$')
plt.xlabel(r'$x$')
plt.legend()
plt.title('Activation function in Neural Networks')
plt.savefig('Activation.png', dpi = 800)
plt.show()
plt.close()









from Utils import *
# import matplotlib.pyplot as plt
# import numpy as np

q = np.arange(8)
T = 2**q

errs = np.arange(0.1, 1, 0.2)
ro = np.arange(0.1, 1, 0.2)

c = 0.8# np.arange(0.1, 1, 0.1)
n = len(errs)*len(ro)
colors = plt.cm.jet(np.linspace(0,5,n*5))
i = 0

map = plt.get_cmap( colors[i] )

for err in errs:
    for c in ro:
        if i< (n-1):
            i+=1
            E_ens = ((1+c*(T-1))/T)*err
            # plt.imshow(q, E_ens, color = colors[i])

            plt.plot(q, E_ens, color = colors[i], cmap = grayscale_map)
        # plt.fill_between(x, y - error, y + error)
plt.show()



err = [1] #np.arange(0.1, 1, 0.05)
ro = np.arange(0.1, 0.9, 0.05)

# c = 0.3 # np.arange(0.1, 1, 0.1)

for c in ro:
    E_ens = ((1+c*(T-1))/T)*err
    plt.plot(q, E_ens)

plt.show()


prova = df.copy()
prova = prova.append(prova)





import numpy as np
import matplotlib.pylab as pl

x = np.linspace(0, 2*np.pi, 64)
y = np.cos(x)

pl.figure()
pl.plot(x,y)

n = 20
colors = pl.cm.jet(np.linspace(0,1,n))

for i in range(n):
    pl.plot(x, i*y, color=colors[i])
pl.show()





# import all functions in Utils
import sys
sys.path.insert(1, '../')

from Utils import *
output = 'output'
create_dir(output)
import numpy as np
import matplotlib.pylab as pl

x = np.linspace(-1, +1, 1000)
y = (1/2) + ((x**2)/2)
z = 1 - ((1/2) + ((x**2)/2))
# z = np.cos(x)
# plt.figure()
plt.plot(x,y)
plt.plot(x,z)
plt.savefig(output + '/cos_cls.png')
plt.show()

n = 20
colors = pl.cm.jet(np.linspace(0,1,n))

for i in range(n):
    pl.plot(x, i*y, color=colors[i])
pl.show()