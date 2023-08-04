
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Population:

    alpha = 0.9     # randomization coefficient
    gamma = 0.5     # light absorbing coefficient
    beta = 0.9      # intensity coefficient

    class Firefly:
        def __init__(self):
            self.pos = np.random.uniform(-10, 10, size=2)

        def __sub__(self, other):
            return self.pos - other.pos

    def __init__(self, count, objective_func, max_gen=5):
        self.count = count;
        self.max_gen = max_gen
        self.fireflies = []
        self.objective_func = objective_func

    def init_generation(self):
        self.fireflies = [Population.Firefly() for _ in range(self.count)]

    def run(self):
        for gen in range(self.max_gen):
            print("Generation: ", gen)

            for i in range(len(self.fireflies)):
                for j in range(len(self.fireflies)):

                    a_intensity = self.objective_func(*self.fireflies[i].pos)
                    b_intensity = self.objective_func(*self.fireflies[j].pos)

                    if (b_intensity > a_intensity):
                        r = np.sqrt(np.sum((self.fireflies[i] - self.fireflies[j]) ** 2))
                        attraction = Population.beta ** np.exp(-Population.gamma * r ** 2)
                        movement = attraction * (self.fireflies[j].pos - self.fireflies[i].pos) + Population.alpha * (np.random.rand(2) - 0.5)
                        self.fireflies[i].pos += movement

        best_index = np.argmax([self.objective_func(*firefly.pos) for firefly in self.fireflies])

        return self.fireflies[best_index]


def objective(a, b):
    return 20 / (3 * a ** 2 + 2 * b ** 2)

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def schaffer_n2(x, y):
    return 0.5 + (np.sin(x**2 - y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2

if __name__ == "__main__":
    p = Population(40, rosenbrock, 10)
    p.init_generation()

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Firefly Algorithm")
    ax = fig.add_subplot(111, projection ='3d')

    x = np.linspace(-50, 50, 500)
    y = np.linspace(-50, 50, 500)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
     
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, alpha=0.8)

    best = p.run()

    print("Best: ", best.pos)
    print("Best: ", objective(*best.pos))

    for firefly in p.fireflies:
        ax.scatter(firefly.pos[0], firefly.pos[1], 0, color="green")

    plt.show()
