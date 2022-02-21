import ea
import ctrnn
import neural_braitV_approach as bv
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import normalize

if len(sys.argv) != 2:
    print("Usage: python rnn_sim.py [neuron-count]")
    exit()


# Nervous System Parameters
N = int(sys.argv[1])       # Number of neurons in the nervous system

if N % 2 != 0 or N == 0:
    print("Neuron count must be even")
    exit()

HN = int(N/2)
S = 2       # Number of sensory inputs into the nervous system
M = 2       # Number of motor outputs from the nervous system

WR = 16     # Weight range: maps from [-1, 1] to: [-16,16]
BR = 16     # Bias range: maps from [-1, 1] to: [-16,16]
SR = 16     # Sensory range: maps from [-1, 1] to: [-16,16]
MR = 1      # Motor range: maps from [-1, 1] to: [-1,1]

# Task parameters
distance = 5                                # Distance between the target and the starting position of the agent
duration = 50                               # Duration of the evaluation
stepsize = 0.1                              # Size of the numerical integration step
time = np.arange(0.0,duration,stepsize)     # Points in time
bearing = np.arange(0.0,2*np.pi,np.pi/4)    # Test the agent on multiple starting angles in relation to the target



# Define fitness function
def fitnessFunction(genotype):
    finaldistance = 0.0
    steps = 0

    for angle in bearing:

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

        # Create the agent body
        body = bv.Agent(N)

        # print("Sensory Weights:\n",sensoryweights)
        # print("Weights:\n",ns.Weights)
        # print("Motor Weights:\n",motorweights)
        # print("Biases:\n",ns.Biases)
        # print("")

        # Create stimuli in environment
        food = bv.Food(distance,angle)

        # Use Euler method to update nervous system and agent body
        for t in time:
            # Set neuron input as the sensor activation levels
            ns.Inputs = np.dot(body.sensor_state(),sensoryweights)
            # Update the nervous system based on inputs
            ns.step(stepsize)
            # Update the body based on nervous system activity
            motorneuron_outputs = np.dot(ns.Outputs, motorweights)
            body.step(food, motorneuron_outputs, stepsize)
            finaldistance += body.distance(food)
            steps += 1
    fitness = np.clip(1 - ((finaldistance/steps)/distance),0,1)

    return fitness


# Evolutionary algorithm parameters
popsize = 50
genesize = HN*N + HN + 1*N + HN*2
# Breakdown: Interneuron Weights + Interneuron Biases + Sensor-to-Inter Weights + Inter-to-Motor Weights
recombProb = 0.5
mutatProb = 1/genesize
generations = 100
demeSize = 2

ga = ea.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.saveFitness(f"{N}_evolution.png")
avgfit, bestfit, genotype = ga.fitStats()
print(genotype)
np.save(f"{N}_neuron_genome", genotype)

# Save best one and visualize its behavior
agentpos = np.zeros((len(bearing),len(time),2))
foodpos = np.zeros((len(bearing),2))

# Visualize results
bearing_index = 0
# food at different angles
for angle in bearing:

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

    # Create the agent body
    body = bv.Agent(N)

    # Create stimuli in the environment
    food = bv.Food(distance,angle)
    foodpos[bearing_index] = food.pos()

    j = 0
    for t in time:
        # Set neuron input as the sensor activation levels
        ns.Inputs = np.dot(body.sensor_state(),sensoryweights)
        # Update the nervous system based on inputs
        ns.step(stepsize)
        # Update the body based on nervous system activity
        motorneuron_outputs = np.dot(ns.Outputs, motorweights)
        body.step(food, motorneuron_outputs, stepsize)
        # Store current body position
        agentpos[bearing_index][j] = body.pos().squeeze()
        j += 1
    bearing_index += 1


for i in range(len(bearing)):
    plt.plot(agentpos[i,:,0],agentpos[i,:,1], color=str(i/len(bearing)))
    plt.plot(foodpos[i, 0], foodpos[i, 1],'o', color=str(i/len(bearing)))

plt.savefig(f"{N}_neuron_all.png")
plt.clf()

# food at a specific angle
food_angle = np.random.choice(bearing)
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

# Create the agent body
body = bv.Agent(N)

# Create stimuli in the environment
food = bv.Food(distance,angle)
foodpos = food.pos()

agentpos = np.zeros((len(time),2))

i = 0
NDs = []
for t in time:
    # Set neuron input as the sensor activation levels
    ns.Inputs = np.dot(body.sensor_state(),sensoryweights)
    # Update the nervous system based on inputs
    ns.step(stepsize)
    # Update the body based on nervous system activity
    NDs.append([ns.Outputs])
    # Update the body based on nervous system activity
    motorneuron_outputs = np.dot(ns.Outputs, motorweights)
    body.step(food, motorneuron_outputs, stepsize)
    # Store current body position
    agentpos[i] = body.pos().squeeze()
    i += 1

plt.plot(agentpos[:,0],agentpos[:,1])
plt.plot(foodpos[0], foodpos[1],'o', color="yellow")
plt.savefig(f"{N}_behavior.png")
plt.clf()
NDs = np.array(NDs)
ns.save(f"{N}_neuron_ctrnn")
np.save(f"{N}_neuron_trajectory",NDs)
for i in range(NDs.shape[1]):
    plt.plot(NDs[:,i])
plt.savefig(f"{N}_neural_dynamics.png")
