import evolutionary_algorithm as ea
import ctrnn
import neural_braitV_approach as bv
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from sklearn.preprocessing import normalize
import os
from multiprocessing.pool import Pool

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments to produce agents for Braitenberg task.')
    parser.add_argument('Brain_Sizes', metavar='BS', type=int, nargs='+',
                    help='a list of brain sizes to simulate')
    parser.add_argument('--count', default='1', type=int, help="Number of agents to produce for each brain")
    parser.add_argument('--dir', default='../Data/Run1/', type=str, help="Directory to store generated data")

    args = parser.parse_args()
    return args


def make_brain(genotype, WR=16, BR=16, SR=16, MR=1):
    gs = genotype.shape[0]
    N = int((-5 + (25+8*gs)**(1/2))/2)
    HN = N//2
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

    return ns, sensoryweights, motorweights


# Define fitness function
def fitnessFunction(genotype,distance=5,duration=50,stepsize=0.1):
    # Task parameters
    time = np.arange(0.0,duration,stepsize)     # Points in time
    bearing = np.arange(0.0,2*np.pi,np.pi/4)    # Test the agent on multiple starting angles in relation to the target

    finaldistance = 0.0
    steps = 0

    for angle in bearing:

        # Create the nervous system
        ns, sensoryweights, motorweights = make_brain(genotype)

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


def run_evolution(N,logdir,popsize=2,recombProb=0.5,generations=4,demeSize=2):
    # Evolutionary algorithm parameters
    HN = N//2
    genesize = HN*N + HN + 1*N + HN*2
    # Breakdown: Interneuron Weights + Interneuron Biases + Sensor-to-Inter Weights + Inter-to-Motor Weights
    mutatProb = 1/genesize
    ga = ea.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
    ga.run()
    ga.saveFitness(logdir+f"{N}_evolution.png")
    avgfit, bestfit, genotype = ga.fitStats()
    np.save(logdir+f"{N}_neuron_genome", genotype)
    ns = make_brain(genotype)[0]
    ns.save(logdir+f"{N}_neuron_ctrnn")

def check_folders(folder):
    # checking if the folder
    # exist or not.
    if not os.path.exists(folder):
        # if the folder is not present
        # then create it.
        os.makedirs(folder)


if __name__ == "__main__":
    args = parse_args()
    if args.dir[-1] != "/":
        args.dir = args.dir + "/"

    check_folders(args.dir)
    
    combinations = []
    for agent_id in range(args.count):
        for N in args.Brain_Sizes:
            dirname = args.dir + f"agent_{agent_id}_"
            combinations.append((N,dirname))
    pool = Pool(processes=None)
    pool.starmap(run_evolution, combinations)

    #run_evolution(N,args.dir,popsize=2,generations=3)

    #ns,mw,sw = ctrnn.ctrnn_from_genome(np.load("./tmp8_neuron_genome.npy"))

