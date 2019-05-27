import numpy as np
from matplotlib import pyplot as plt

# Parameters
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