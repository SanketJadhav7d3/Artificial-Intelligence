
import numpy as np

class SigmoidNeuron:
    
    def __init__(self, inputs, outputs):
        # randomly initialize weights and bias
        self.weights = np.random.randint(shape=(inputs.shape[1]))
        self.bias = np.random.randint()
