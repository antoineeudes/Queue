import numpy as np
import random
import math

delta = 0.5



def step_trajectory(x, A=None):
    if A==None:
        A = createA()
    u = random.rand()
    prob = A[x,:]
    cumulated_density = np.cumsum(prob)
    next_state = 0
    while cumulated_density[next_state]<u:
        next_state +=1
    return next_state


def trajectory(x0, t, A=None):
    Xt = np.zeros(t)
    for i in range(math.floor(1./delta*t)):
        Xt[i] = x
        x = step_trajectory(x, A)
    return Xt




