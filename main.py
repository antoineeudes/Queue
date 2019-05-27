import numpy as np
from matplotlib import pyplot as plt
import random
import math
from numpy import linalg as LA

# Parameters
delta = 0.5
K = 1
lbd = 1.
mu = 2.
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

def plot_Xt(T):
    Xt, time = trajectory(T)
    X, Y = [], []
    X.append(Xt[0])
    Y.append(time[0])
    Y.append(time[1])
    for i in range(1, len(Xt)-1):
        X.append(Xt[i-1])
        X.append(Xt[i])
        Y.append(time[i])
        Y.append(time[i+1])
    X.append(Xt[-1])
    plt.plot(Y, X)
    plt.xlabel('t')
    plt.ylabel('Nombre de clients')
    plt.show()


def indicatrice(a, xt):
    if a==xt:
        return 1
    return 0

def density_Xt(x, t=1000):
    Xt, time = trajectory(t)
    s = 0.
    for i in range(len(Xt)-1):
        duration = time[i+1] - time[i]
        s += duration * Xt[i] * indicatrice(Xt[i], x)
    return s/t

# density_vect = np.vectorize(density_Xt, excluded=['t'])

# def plot_density():
#     x = np.linspace(0, 10, 100)
#     y = density_vect(x)
#     print(y)
#     plt.plot(x, y)
#     plt.show()




if __name__ == '__main__':
    print(A)
    plot_Xt(10000)
    # plot_density()
