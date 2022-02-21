import numpy as np                      #Python numerical computational library 
import ctrnn                            #Python Library for simulating CTRNNs (taken from class github)
import matplotlib.pyplot as plt         #Python library for graphing 
import pandas as pd                     #Python library for data manipulation
import neural_braitV_approach as w      #Python library for breitberg task (adapted from Neil Ni)
import feedforward                      #Python library for feedforward neural networks
import pyinform                         #Python library for information theory measures
from mpl_toolkits.mplot3d import Axes3D #Python toolkit for circle plotting

def get_input_vectors(distance,extent=-1):
    r = 1
    o = np.pi/2
    a = np.pi/8
    angles = np.linspace(0,np.pi,181)
    x_coordinates = np.cos(angles) * distance
    y_coordinates = np.sin(angles) * distance
    s1x = r * np.cos(o + a)     # sensor 1 x position
    s1y = r * np.sin(o + a)     # sensor 1 y position
    s2x = r * np.cos(o - a)     # sensor 2 x position
    s2y = r * np.sin(o - a)     # sensor 2 y position
    d1 = np.sqrt((x_coordinates-s1x)**2+(y_coordinates-s1y)**2)
    d2 = np.sqrt((x_coordinates-s2x)**2+(y_coordinates-s2y)**2)
    inputs = np.stack((d1,d2),1)
    inputs = np.concatenate((inputs,np.flip(inputs,axis=1)),0)
    inputs = np.concatenate((inputs,inputs),0)
    inputs = np.concatenate((inputs,inputs),0)
    return inputs[:extent]

def get_agent(n=2):
    agent = ctrnn.CTRNN(n)
    agent.load(f"./Run2/{n}_neuron_ctrnn.npz")
    a = np.load(f"./Run2/{n}_neuron_genome.npy")
    count = n/2 * 2
    mw = a[-n:]
    mw = np.stack((mw,np.flip(mw)),axis=1)
    sw = a[-2*n:-n] * 16
    sw = np.stack((sw,np.flip(sw)))
    return agent, mw, sw

def get_agent_ff(n=2):
    agent = feedforward.FeedForwardNetwork(n)
    agent.load(f"./{n}_neuron_ff.npz")
    a = np.load(f"./{n}_neuron_genome.npy")
    return agent

def tease_agents(distances=[2],ns=[2,4,8,16,32],wait=False):
    fig, axs = plt.subplots(len(ns), sharex=True, sharey=False, figsize=(20,10))
    fig.suptitle('Neural Dynamics')
    for i in range(1,1+len(ns)):
        n = 2 ** i
        agent, mw, sw = get_agent(n)
        N = 2
        HN = int(N/2)
        gs = HN * N + HN + N + HN
        alpha_index = 0

        for distance in distances:
            inputs = get_input_vectors(distance)

            activations = []
            body = w.Agent(np.ones(gs))
            food = w.Food(0,0)
            agent.initializeState(0)
            for j in range(100):
                agent.step(0.01)
            for inp in inputs:
                agent.Inputs = np.dot(inp,sw)
                agent.step(0.01)
                if wait:
                    for j in range(200):
                        agent.step(0.01)
                motorout = np.dot(agent.Outputs,mw)
                body.step(food, motorout, 0.01)
                activations.append(motorout)
            activations = np.array(activations)
            max_speed = np.dot(np.ones(n),mw).sum()
            axs[i-1].plot(activations[:,0],color="blue", label=f"Motor 1-Distance {distance}",alpha=1-alpha_index)
            axs[i-1].plot(activations[:,1],color="orange",label=f"Motor 2-Distance {distance}",alpha=1-alpha_index)
            #axs[i-1].set_ylim([activations[50:].min(),activations[50:].max()])
            alpha_index += 0.3
        if i == 3:
            axs[i-1].set_ylabel("Motor Outputs")
        axs[i-1].set_title(f"{n}-Neurons")

    plt.legend(loc="lower left")
    plt.show()
