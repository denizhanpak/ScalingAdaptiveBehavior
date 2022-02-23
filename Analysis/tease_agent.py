
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
