
import numpy as np


class SigmoidNeuron:
    
    def __init__(self):
        # randomly initialize weights and bias
        self.weight = np.random.randn()
        self.bias = np.random.randn()

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

    def gradient_descent(self, inputs, outputs, max_epochs=1000, print_diagnostic=True):
        lr = 0.1

        for i in range(max_epochs):
            dw, db = 0, 0

            for x, y in zip(inputs, outputs):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)

            self.weight = self.weight - lr * dw
            self.bias = self.bias - lr * db

            if print_diagnostic:
                if i % 100 == 0:
                    print("Error: ", self.error(inputs, outputs))

    def stochastic_gradient_descent(self, inputs, outputs, max_epochs=1000, print_diagnostic=True):
        lr = 0.1

        for i in range(max_epochs):
            dw, db = 0, 0

            for x, y in zip(inputs, outputs):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)

                self.weight = self.weight - lr * dw
                self.bias = self.bias - lr * db

            if print_diagnostic:
                if i % 100 == 0:
                    print("Error: ", self.error(inputs, outputs))

    def mini_batch_stochastic_gradient_descent(self, inputs, outputs, 
                                               lr=0.1, batch_size=10, max_epochs=1000, print_diagnostic=True):

        for i in range(max_epochs):
            dw, db = 0, 0
            points_seen = 0

            for x, y in zip(inputs, outputs):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
                points_seen += 1

                if points_seen % batch_size == 0:
                    self.weight = self.weight - lr * dw
                    self.bias = self.bias - lr * db
                    dw, db = 0, 0

            if print_diagnostic and i % 100 == 0:
                    print("Error: ", self.error(inputs, outputs))
