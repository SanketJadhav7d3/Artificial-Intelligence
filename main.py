
from lib.tensor import Tensor
from lib.deriv.derivative import deriv
from lib.sigmoid_neuron import SigmoidNeuron
import numpy as np
import matplotlib.pyplot as plt

X = [0.5, 3.4]
Y = [0.4, 0.9]

s = SigmoidNeuron()

s.train(X, Y)

plt.scatter(X, Y)

x_lin = np.arange(-10, 10, 0.1)

plt.plot(x_lin, s.sigmoid(x_lin))

plt.show()
