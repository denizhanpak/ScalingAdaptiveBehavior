
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
