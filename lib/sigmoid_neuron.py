
import numpy as np


class SigmoidNeuron:
    
    def __init__(self):
        # randomly initialize weights and bias
        self.weight = 4
        self.bias = np.random.sample()

    def sigmoid(self, x: int) -> float:
        return 1 / (1 + np.exp(-(np.dot(x, self.weight) + self.bias)))

    def error(self, X, Y):
        # squared error
        err = 0.0
        for x, y in zip(X, Y):
            fx = self.sigmoid(x)
            err += (fx - y) ** 2
        return err

    def grad_b(self, x, y):
        fx = self.sigmoid(x)
        return (fx - y) * fx * (1 - fx)

    def grad_w(self, x, y):
        fx = self.sigmoid(x)
        return (fx - y) * fx * (1 - fx) * x

    def train(self, inputs, outputs, max_epochs=1000, print_diagnostic=True):
        lr = 0.1

        for i in range(max_epochs):
            dw, db = 0, 0

            for x, y in zip(inputs, outputs):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)

            if print_diagnostic:
                if i % 100 == 0:
                    print("Error: ", self.error(inputs, outputs))

            self.weight = self.weight - lr * dw
            self.bias = self.bias - lr * db
