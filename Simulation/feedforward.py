import numpy as np
import matplotlib.pyplot as plt
from torch import nn

def np_relu(a):
    a[a < 0] = 0
    return a

def np_sig(a):
    return 1/(1 + np.exp(-a))

def identity(a):
    return a

class FeedForwardNetwork():

    def __init__(self, size, non_linearity=np_relu):
        self.Size = size                        # number of neurons in the circuit
        self.Weights = np.zeros((2,size))       # connection weight for input to hidden layer
        self.Outputs = np.zeros((size, 2))      # connection weight for hidden layer to output
        self.nl = non_linearity

    def setWeights(self, weights, out_weights):
        self.Weights = weights
        self.Outputs = out_weights

    def randomizeParameters(self):
        self.Weights = np.random.uniform(-10,10,size=(2, self.Size))
        self.Outputs = np.random.uniform(-10,10,size=(self.Size, 2))

    def step(self, obs):
        assert obs.shape == (2,)
        yhat = self.nl(np.dot(obs, self.Weights))
        y = np_sig(np.dot(yhat, self.Outputs)) * 2 - 1

        return y

    def save(self, filename):
        np.savez(filename, size=self.Size, weights=self.Weights, outputs=self.Outputs)

    def load(self, filename):
        params = np.load(filename)
        self.Size = params['size']
        self.Weights = params['weights']
        self.Outputs = params['outputs']
