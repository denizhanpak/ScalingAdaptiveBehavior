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

def get_agent(n=2, folder="./Run2/", agent_id=None):
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

