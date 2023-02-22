
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import numpy as np
from lib.sigmoid_neuron import SigmoidNeuron


ws = np.arange(-5, 5, 0.1)
bs = np.arange(-5, 5, 0.1)

# find sigmoid function which passes through both points
X = [0.35, 0.95, 1.56]
Y = [0.35, 0.35, 0.35]

def sigmoid(x: list, w: list, b: float):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))

def calculate_error(x, w, b):
    # composite function
    err = 0.0
    
    for x, y in zip(X, Y):
        fx = sigmoid(x, w, b)
        err += (fx - y) ** 2
        
    return err

# train the model
s = SigmoidNeuron()


w_mesh, b_mesh = np.meshgrid(ws, bs)

error = np.zeros(w_mesh.shape)


fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
ax.set_zlabel("Error")
ax.set_xlabel("W")
ax.set_ylabel("B")

for i in range(100):
    for j in range(100):
        error[i][j] = calculate_error(0.1, w_mesh[i][j], b_mesh[i][j])

ax.plot_surface(w_mesh, b_mesh, error, cmap=cm.coolwarm, linewidth=0, alpha=0.7)

weights = []
biases = []
errors = []

for i in range(5000):

    s.train(X, Y, 1)

    if i % 50 == 0:
        ax.scatter(s.weight, s.bias, calculate_error(1, s.weight, s.bias), color='green')

        weights.append(s.weight)
        biases.append(s.bias)
        errors.append(s.error(X, Y))

plt.plot(weights, biases, errors, 'gray')

plt.show()
