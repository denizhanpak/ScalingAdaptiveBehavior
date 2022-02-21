import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class CTRNN():

    def __init__(self, size):
        self.Size = size                        # number of neurons in the circuit
        self.States = np.zeros(size)            # state of the neurons
        self.TimeConstants = np.ones(size)      # time-constant for each neuron
        self.invTimeConstants = 1.0/self.TimeConstants
        self.Biases = np.zeros(size)            # bias for each neuron
        self.Weights = np.zeros((size,size))    # connection weight for each pair of neurons
        self.Outputs = np.zeros(size)           # neuron outputs
        self.Inputs = np.zeros(size)            # external input to each neuron

    def setWeights(self, weights):
        self.Weights = weights

    def setBiases(self, biases):
        self.Biases = biases

    def setTimeConstants(self, timeconstants):
        self.TimeConstants = timeconstants
        self.invTimeConstants = 1.0/self.TimeConstants

    def randomizeParameters(self):
        self.Weights = np.random.uniform(-10,10,size=(self.Size,self.Size))
        self.Biases = np.random.uniform(-10,10,size=(self.Size))
        self.TimeConstants = np.random.uniform(0.1,5.0,size=(self.Size))
        self.invTimeConstants = 1.0/self.TimeConstants

    def initializeState(self, s):
        self.States = s
        self.Outputs = sigmoid(self.States+self.Biases)

    def step(self, dt):
        netinput = self.Inputs + np.dot(self.Weights.T, self.Outputs)
        self.States += dt * (self.invTimeConstants*(-self.States+netinput))
        self.Outputs = sigmoid(self.States+self.Biases)

    def save(self, filename):
        np.savez(filename, size=self.Size, weights=self.Weights, biases=self.Biases, timeconstants=self.TimeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.Size = params['size']
        self.Weights = params['weights']
        self.Biases = params['biases']
        self.TimeConstants = params['timeconstants']
        self.invTimeConstants = 1.0/self.TimeConstants

def ctrnn_from_genome(genotype):
    HN = int(N/2)
    WR = 16     # Weight range: maps from [-1, 1] to: [-16,16]
    BR = 16     # Bias range: maps from [-1, 1] to: [-16,16]
    SR = 16     # Sensory range: maps from [-1, 1] to: [-16,16]
    MR = 1      # Motor range: maps from [-1, 1] to: [-1,1]
    
    # Create the nervous system
    ns = ctrnn.CTRNN(N)
    
    # Set the parameters of the nervous system according to the genotype-phenotype map
    k = 0
    weights = np.zeros((N,N))
    for i in range(HN):
        for j in range(N):
            weights[i,j] = genotype[k]
            weights[N-i-1,N-j-1] = genotype[k]
            k += 1
    ns.setWeights(weights * WR)

    biases = np.zeros(N)
    for i in range(HN):
        biases[i] = genotype[k]
        biases[N-i-1] = genotype[k]
        k += 1
    ns.setBiases(biases * BR)

    ns.setTimeConstants(np.array([1.0]*N))

    sensoryweights = np.zeros((2,N))
    for j in range(N):
        sensoryweights[0,j] = genotype[k]
        sensoryweights[1,N-j-1] = genotype[k]
        k += 1
    sensoryweights = sensoryweights * SR

    motorweights = np.zeros((N,2))
    for i in range(HN):
        for j in range(2):
            motorweights[i,j] = genotype[k]
            motorweights[N-i-1,2-j-1] = genotype[k]
            k += 1
    motorweights = motorweights * MR
    motorweights = normalize(motorweights,axis=0,norm='l2') 

    # Initialize the state of the nervous system to some value
    ns.initializeState(np.zeros(N))

    return 
