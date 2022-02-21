import ea
from feedforward import FeedForwardNetwork
import neural_braitV_approach as bv
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import normalize

# Nervous System Parameters
N = int(sys.argv[1])       # Number of neurons in the nervous system
sw_count = N * 3  // 2     # Number of sensor weights after symmetry
mw_count = N               # Number of motor weights after symmetry 
HN = N // 2

WR = 1     # Weight range: maps from [-1, 1] to: [-16,16]

# Task parameters
genotype = np.array(range(1,sw_count+1+mw_count))
distance = 5                                # Distance between the target and the starting position of the agent
duration = 50                               # Duration of the evaluation
stepsize = 0.1                              # Size of the numerical integration step
time = np.arange(0.0,duration,stepsize)     # Points in time
bearing = np.arange(0.0,2*np.pi,np.pi/4)    # Test the agent on multiple starting angles in relation to the target

def make_agent(genotype):
    genotype *= WR
    k = 0
    sw = np.zeros((2,N))
    mw = np.zeros((N,2))
    for i in range(2):
        for j in range(HN):
            sw[i,j] = genotype[k]
            sw[2-i-1,N-j-1] = genotype[k]
            k += 1
    for i in range(HN):
        for j in range(2):
            mw[i,j] = genotype[k]
            mw[N-i-1,2-j-1] = genotype[k]
            k += 1
    ns = FeedForwardNetwork(N)
    ns.setWeights(sw,mw)
    return ns


# Define fitness function
def fitnessFunction(genotype):
    finaldistance = 0.0
    steps = 0

    for angle in bearing:

        # Create the nervous system
        ns = make_agent(genotype)

        # Create the agent body
        body = bv.Agent(N)

        # Create stimuli in environment
        food = bv.Food(distance,angle)

        # Use Euler method to update nervous system and agent body
        for t in time:
            # Set neuron input as the sensor activation levels
            out = ns.step(body.sensor_state())

            # Update the body based on nervous system activity
            body.step(food, out, stepsize)
            finaldistance += body.distance(food)
            steps += 1

    fitness = np.clip(1 - ((finaldistance/steps)/distance),0,1)

    return fitness


# Evolutionary algorithm parameters
popsize = 20
genesize = sw_count+mw_count
# Breakdown: Interneuron Weights + Interneuron Biases + Sensor-to-Inter Weights + Inter-to-Motor Weights
recombProb = 0.5
mutatProb = 1/genesize
generations = 50
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
    ns = make_agent(genotype)

    # Create the agent body
    body = bv.Agent(N)

    # Create stimuli in the environment
    food = bv.Food(distance,angle)
    foodpos[bearing_index] = food.pos()

    j = 0
    for t in time:
        # Set neuron input as the sensor activation levels
        out = ns.step(body.sensor_state())
        
        # Update the body based on nervous system activity
        body.step(food, out, stepsize)
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
ns = make_agent(genotype)
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
    out = ns.step(body.sensor_state())
    NDs.append(out)
    
    # Update the body based on nervous system activity
    body.step(food, out, stepsize)
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
