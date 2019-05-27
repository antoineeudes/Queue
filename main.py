import numpy as np
import random
import math
from numpy import linalg as LA

# Parameters
delta = 0.5
K = 1
lbd = 1.
mu = 1.
N_max = 10

def createA(n):
    A = np.zeros((n, n))
    for i in range(n-1):
        A[i, i+1] = lbd
        A[i+1, i] = min(i+1, K)*mu
    for i in range(n):
        A[i, i] = -(lbd + min(i, K)*mu)
        
    return A


def step_trajectory(x):
    u = random.rand()
    prob = A[x,:]
    cumulated_density = np.cumsum(prob)
    next_state = 0
    while cumulated_density[next_state]<u:
        next_state +=1
    return next_state


def trajectory(t, x0=0):
    Xt = np.zeros(int(np.floor(t/delta)))
    x = x0
    for i in range(int(np.floor(t/delta))):
        Xt[i] = x
        if x == 0:
            x = 1
        else:
            u = random.random()
            if u < lbd/(lbd+mu):
                x += 1
            else:
                x -= 1
    return Xt





A = createA(N_max) 

print(trajectory(5))