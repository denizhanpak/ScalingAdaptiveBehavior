import numpy as np

class Agent:

    def __init__(self, can_reverse=False):
        self.x = 0.0                                    # agent's x position
        self.y = 0.0                                    # agent's y position
        self.o = 0.0                                    # agent's orientation
        self.v = 0.5                                    # agent's velocity
        self.r = 1.0                                    # agent's radius
        self.a = np.pi/8                                # sensor angle offset
        self.s1 = 0.0                                   # sensor 1 value
        self.s2 = 0.0                                   # sensor 2 value

        self.s1x = self.r * np.cos(self.o + self.a)     # sensor 1 x position
        self.s1y = self.r * np.sin(self.o + self.a)     # sensor 1 y position
        self.s2x = self.r * np.cos(self.o - self.a)     # sensor 2 x position
        self.s2y = self.r * np.sin(self.o - self.a)     # sensor 2 y position

        self.cr = can_reverse

    def sensor_state(self):
        return np.array([self.s1, self.s2]).squeeze()#, self.s1, self.s4]).squeeze()

    def step(self, food, motorneuron_output, stepsize):
        # Sense: Calculate the distance of the pizza to the sensors
        self.s1 = 1/np.sqrt((self.s1x-food.x)**2 + (self.s1y-food.y)**2)
        self.s2 = 1/np.sqrt((self.s2x-food.x)**2 + (self.s2y-food.y)**2)

        ## Some transformation between nervous system to the left and right motor neurons
        self.rmn = motorneuron_output[0]
        self.lmn = motorneuron_output[1]

        if not self.cr:
            if self.lmn < 0:
                self.lmn = 0
            if self.rmn < 0:
                self.rmn = 0

        # Translate from right and left motor to orientation and velocity
        self.o += 1*(self.rmn - self.lmn)
        self.v = self.rmn + self.lmn

        # Move: Update position of the agent using Euler Method
        self.x += stepsize * (self.v*np.cos(self.o))
        self.y += stepsize * (self.v*np.sin(self.o))

        # Remember to update position of the sensors as well!
        self.s1x = self.x + self.r * np.cos(self.o + self.a)
        self.s1y = self.y + self.r * np.sin(self.o + self.a)
        self.s2x = self.x + self.r * np.cos(self.o - self.a)
        self.s2y = self.y + self.r * np.sin(self.o - self.a)

    def distance(self, food):
        return np.sqrt((self.x-food.x)**2 + (self.y-food.y)**2)

    def pos(self):
        return np.array([self.x, self.y])
    
    def orient(self):
        return np.array([self.o])

class Food:

    def __init__(self, dist, angle):
        self.x = dist * np.cos(angle)
        self.y = dist * np.sin(angle)

    def pos(self):
        return np.array([self.x, self.y])
