import numpy as np
import random
import math
from numpy import linalg as LA

# Parameters
delta = 0.5
K = 1
lbd = 1
mu = 1
N_max = 10

def createA(n):
    A = np.zeros((n, n))
    for i in range(n-1):
        A[i, i+1] = lbd
        A[i+1, i+2] = min(i+1, K)*mu
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


def trajectory(x0, t):
    Xt = np.zeros(math.floor(t/delta))
    incoming_queue = np.random.exponential(lbd, np.floor(t/delta))
    out_queue = np.random.exponential(lbd, np.floor(t/mu))
    for i in range(math.floor(1./delta*t)):
        Xt[i] = x
        x = x + incoming_queue[i] - out_queue[i]
    return Xt





A = createA(n_max) 
