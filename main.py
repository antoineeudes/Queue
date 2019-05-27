import numpy as np
from matplotlib import pyplot as plt
import random
import math
from numpy import linalg as LA

# Parameters
delta = 0.5
K = 1
lbd = 1
mu = 1

T_max = 10
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


def trajectory(T, x0=0):
    Xt = np.array([])
    t = 0
    x = x0
    time = np.array([])
    while t<T:
        Xt = np.append(Xt, x)
        time = np.append(time, t)
        if x == 0:
            x = 1
            t += np.random.exponential(lbd)
        else:
            u = random.random()
            if u < lbd/(lbd+mu):
                x += 1
            else:
                x -= 1
            t += np.random.exponential(lbd+mu)
    return Xt, time


A = createA(N_max) 

def plot_Xt(Xt):
    n, = Xt.shape
    X = delta*np.arange(n)
    plt.plot(X, Y)
    plt.xlabel('t')
    plt.ylabel('Nombre de clients')
    plt.show()


if __name__ == '__main__':
    print(A)