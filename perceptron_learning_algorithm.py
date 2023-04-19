
import matplotlib.pyplot as plt
import numpy as np
import math

# AND Boolean function
Xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
ys = np.array([0, 0, 0, 1])

Ws = np.random.sample((2,))

bias = -1

def angle(a, b):
    mod_a = math.sqrt(np.dot(a, a))
    mod_b = math.sqrt(np.dot(b, b))
    return np.arccos(np.dot(a, b) / (mod_a * mod_b))

def learn(epochs=5):
    global Ws

    for epoch in range(epochs):
        print("Epoch: ", epoch+1)
        for X, y in zip(Xs, ys):
            if y == 1 and np.dot(Ws, X) + bias < 0:
                Ws += X
            elif y == 0 and np.dot(Ws, X) + bias >= 0:
                Ws -= X

    print("Optimal Ws: ", Ws)

def test():
    print("Test: ")
    for X, y in zip(Xs, ys):
        if np.dot(Ws, X) + bias >= 0:
            print("1", np.dot(Ws, X) + bias)
        else:
            print("0", np.dot(Ws, X) + bias)

learn()
test()

# plot the result
plt.scatter(Xs[:, 0], Xs[:, 1])
plt.plot(Xs[1:3, 0], Xs[1:3, 1], linestyle="dotted")
plt.plot(Ws + bias, label="Weight vector")

# calculate angle between wegiht vector and inputs
print(np.rad2deg(angle(Ws + bias, Xs[1])))
print(np.rad2deg(angle(Ws + bias, Xs[2])))
print(np.rad2deg(angle(Ws + bias, Xs[3])))

plt.legend()
plt.show()
