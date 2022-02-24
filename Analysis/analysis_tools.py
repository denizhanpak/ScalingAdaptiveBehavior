import numpy as np                      #Python numerical computational library 
import ctrnn                            #Python Library for simulating CTRNNs (taken from class github)
import matplotlib.pyplot as plt         #Python library for graphing 
import pandas as pd                     #Python library for data manipulation
#import neural_braitV_approach as w      #Python library for breitberg task (adapted from Neil Ni)
#import feedforward                      #Python library for feedforward neural networks
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

def get_agent(n=2, folder="../Data/Run1/", agent_id=0):
    agent = ctrnn.CTRNN(n)
    agent.load(folder+f"agent_{agent_id}_{n}_neuron_ctrnn.npz")
    a = np.load(folder+f"agent_{agent_id}_{n}_neuron_genome.npy")
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

def MultiQuasiStaticApproximation(agent,multiinputs,steps=500):#,folder=):
    brain, mw, sw = agent
    trajectories = []
    rv = []
    
    for j, inputs in enumerate(multiinputs):
        rv.append([])
        for inp in inputs:
            rv[-1].append([])
            activations = []
            brain.Inputs = np.dot(inp,sw)
            brain.initializeState(0)
            for i in range(steps):
                brain.step(0.01)
                motorout = np.dot(brain.Outputs,mw)
                rv[-1][-1].append(motorout)
                activations.append(motorout)
            activations = np.array(activations)
            
    return np.array(rv)

def plot_mqsa(responses):
    fig, axs = plt.subplots(responses.shape[0], sharex=True, sharey=False, figsize=(20,10))
    for distance, angle_vec in enumerate(responses):
        alpha_index = 0
        for angle,traj in enumerate(angle_vec):
            axs[distance].plot(traj[:,0],alpha=1-0.99*alpha_index,color="cyan")
            axs[distance].plot(traj[:,1],alpha=1-0.99*alpha_index,color="magenta")
            alpha_index += 1/responses.shape[1]
        axs[distance].set_ylabel("Neuron Outputs")
        axs[distance].set_xlabel("Time")

    plt.show()

def plot_av_mqsa(responses,agent_name="Bob"):
    fig, axs = plt.subplots(responses.shape[0], sharex=True, sharey=False, figsize=(20,10))
    for distance, angle_vec in enumerate(responses):
        alpha_index = 0
        for angle,traj in enumerate(angle_vec):
            l = np.abs(traj[:,0] - traj[:,1])
            axs[distance].plot(l,alpha=1-0.99*alpha_index,color="springgreen")
            alpha_index += 1/responses.shape[1]
        axs[distance].set_ylabel("Produced Angular Velocity")
        axs[distance].set_xlabel("Time")

    plt.savefig(f"../Data/{agent_name}_angular_velocity.png")
    plt.clf()

def get_transience(timeseries,dx=0.01):
    lim = 0.000001
    # Iterate through time series and identify equillibria
    prev_point = -1
    counter = 0
    for point in timeseries:
        if np.abs(point - prev_point) < 0.0001:
            break
        prev_point = point
        counter += 1
    
    #An equillibrium is never reached
    if counter == timeseries.shape[0]:
        return lim
    
    transient = timeseries[:counter]
    
    transient_volume = np.maximum(lim,np.trapz(transient,dx=dx))
    equillibrium_volume = dx * counter * point
    rv = equillibrium_volume/transient_volume
    #rv = np.log(rv)
    return rv

def get_all_transiences(trajs,dx=0.0001):
    rv = []
    for di, trials in enumerate(trajs):
        rv.append([])
        for ai, run in enumerate(trials):
            t = np.abs(run[:,0] - run[:,1])
            rv[-1].append(get_transience(t))
    rv = np.array(rv)
    return rv

def plot_agent_circle_transience(transiences,ci=-1,agent_name="Bob"):
    colors = ["Wistia","YlGnBu_r","hot","summer","winter","cool","bone","YlOrBr","autumn","plasma","inferno"]
    circle_plot(transiences,cmap=colors[ci])
    plt.title(f"Intransience of Orientation Response for Agent: {agent_name}")
    plt.savefig(f"../Data/{agent_name}_transience.png")
    plt.clf()

def circle_plot(data,cmap="PiYG"):
    #data = np.stack((data,data),axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    print(data.shape)
    if len(data.shape) == 1:
        m = 2
        n = data.shape[0] + 1
        
    else:
        data = np.swapaxes(data,0,1)
        m = data.shape[1] + 1
        n = data.shape[0] + 1
        
    rad = np.linspace(0, data.shape[1], m)
    a = np.linspace(0, 2 * np.pi, n)
    r, th = np.meshgrid(rad, a)

    z = np.random.uniform(-1, 1, (n,m))
    #print(z)
    plt.subplot(projection="polar")

    plt.pcolormesh(th, r, data, cmap=cmap)

    plt.plot(a, r, ls='none', color = 'k') 
    plt.grid()
    plt.colorbar()


mis = []
for i in [1,2,3,4,5]:
    inputs = get_input_vectors(i,extent=360)
    mis.append(inputs)

for n in [2,4,8,16,32]:
    rv = MultiQuasiStaticApproximation(get_agent(agent_id=0, n=n), mis)
    for aid in range(1,100):
        rv += MultiQuasiStaticApproximation(get_agent(agent_id=aid, n=n), mis)

    rv /= 100

    plot_av_mqsa(responses=rv[:,:91,:,:],agent_name=f"brainsize_{n}_run1_avg")
    trajs = get_all_transiences(rv)
    trajs /= trajs.max()
    plot_agent_circle_transience(trajs,agent_name=f"brainsize_{n}_run1_avg")