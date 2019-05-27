import numpy as np
from matplotlib import pyplot as plt
import random
import math

# Parameters
delta = 0.5
K = 1
lbd = 1
mu = 1

T_max = 10
N_max = 10

def create_A(n):
    A = np.zeros((n, n))
    for i in range(n-1):
        A[i, i+1] = lbd
        A[i+1, i] = min(i+1, K)*mu
    for i in range(n):
        A[i, i] = -(lbd + min(i, K)*mu)
        
    return A


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




A = createA(N_max)
def plot_Xt(Xt):
    n, = Xt.shape
    X = delta*np.arange(n)
    plt.plot(X, Y)
    plt.xlabel('t')
    plt.ylabel('Nombre de clients')
    plt.show()

A = create_A(N_max)

if __name__ == '__main__':
    print(A)