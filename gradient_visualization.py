
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import numpy as np
from lib.sigmoid_neuron import SigmoidNeuron


ws = np.arange(-5, 5, 0.1)
bs = np.arange(-5, 5, 0.1)

# find sigmoid function which passes through both points
X = [0.1, 0.5, 0.9]
Y = [0.2, 0.1, 0.5]

def sigmoid(x: list, w: list, b: float):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))

def calculate_error(x, w, b):
    # composite function
    err = 0.0
    
    for x, y in zip(X, Y):
        fx = sigmoid(x, w, b)
        err += (fx - y) ** 2
        
    return err

s = SigmoidNeuron()

w_mesh, b_mesh = np.meshgrid(ws, bs)

error = np.zeros(w_mesh.shape)

fig = plt.figure(figsize=(10, 8))
fig.suptitle("Gradient Visualization")
ax = fig.add_subplot(2, 1, 1, projection ='3d')
 
ax.set_zlabel("Error")
ax.set_xlabel("W")
ax.set_ylabel("B")

for i in range(100):
    for j in range(100):
        error[i][j] = calculate_error(0.1, w_mesh[i][j], b_mesh[i][j])

ax.plot_surface(w_mesh, b_mesh, error, cmap=cm.coolwarm, linewidth=0, alpha=0.8)

weights = [s.weight]
biases = [s.bias]
errors = [s.error(X, Y)]

ax.scatter(s.weight, s.bias, calculate_error(1, s.weight, s.bias), color='green')

# training the model
for i in range(4000):

    s.do_adagrad(X, Y, 1)

    if i % 50 == 0:
        ax.scatter(s.weight, s.bias, calculate_error(1, s.weight, s.bias), color='green')

        weights.append(s.weight)
        biases.append(s.bias)
        errors.append(s.error(X, Y))

plt.plot(weights, biases, errors, 'gray')

ax = fig.add_subplot(2, 1, 2)

ax.contourf(w_mesh, b_mesh, error, cmap=cm.coolwarm)

ax.scatter(weights, biases, c='b', s=4)
ax.plot(weights, biases, c='g')

ax.set_xlabel("Weight")
ax.set_ylabel("Bias")

plt.show()

# fig.savefig("images/gradient_visualization.png", bbox_inches='tight')
