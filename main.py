import numpy as np

# Parameters
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

A = createA(N_max)
