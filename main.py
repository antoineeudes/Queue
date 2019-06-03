import numpy as np
from matplotlib import pyplot as plt
import random
import math
from numpy import linalg as LA

# Parameters
delta = 0.5
K = 1
lbd = 0.7
mu = 1.
rho = lbd/mu

T_max = 10
N_max = 100

def createA(n, lbd=lbd, mu=mu):
    '''
        Create the infinitesimal generator with shape (n, n)
    '''
    A = np.zeros((n, n))

    for i in range(n-1):
        A[i, i+1] = lbd
        A[i+1, i] = min(i+1, K)*mu
    for i in range(n):
        A[i, i] = -(lbd + min(i, K)*mu)

    return A

def createP(n, lbd=lbd, mu=mu):
    '''
        Create the transition matrix of the Xn Markov Chain
    '''
    P = np.zeros((n, n))

    for i in range(n):
        for j in range(min(i+2, n)):
            P[i, j] = mu**(i-j+1)/(lbd+mu)**(i-j+1)*(int(j==0)+lbd/(lbd+mu)*(int(j>0)))

    return P

A = createA(N_max)

def trajectory(T, x0=0, lbd=lbd, mu=mu):
    '''
        Simulate a trajectory for Xt starting x0 with horizon T
    '''
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



def plot_Xt(T, lbd=lbd, mu=mu):
    '''
        Simulate and plot a trajectory of Xt with horizon T
    '''
    Xt, time = trajectory(T, lbd=lbd, mu=mu)
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
    plt.title('Nombre de clients en fonction du temps avec lbd={} et mu={}'.format(lbd, mu))
    plt.show()

def compute_pi(n=N_max, lbd=lbd, mu=mu):
    '''
        Solve xA=0 where x is normalized
    '''
    A = createA(n, lbd=lbd, mu=mu)

    # To avoid nul solution
    # force last component to be 1
    A = np.transpose(A)
    b = -A[:-1, -1]
    A = A[:-1, :-1]

    pi = np.linalg.solve(A, b)
    pi = np.append(pi, 1)

    return pi/np.sum(pi)

def plot_pi(n):
    pi = compute_pi(n=n)
    X = np.arange(n)
    plt.plot(X, pi, c='orange')
    plt.xlabel('Nombre de clients')
    plt.ylabel('Probabilité')
    plt.title('Densité de la probabilité invariante pi')
    plt.show()

def estimate_exp_var(x0=0, n=N_max, lbd=lbd, mu=mu, show=False):
    '''
        Estimate and return the expectancy and variance of Xt
        Return also the expected expectancy and variance
    '''
    pi = compute_pi(n=n, lbd=lbd, mu=mu)
    estimated_exp = np.vdot(pi, np.arange(n))
    estimated_var = np.vdot(pi, [(x-estimated_exp)**2 for x in range(n)])
    expected_exp = rho/(1-rho)
    expected_var = rho/(1-rho)**2

    if show:
        print('Estimated expectancy : {}'.format(estimated_exp))
        print('Expected expectancy : {}'.format(expected_exp))
        print('Estimated variance : {}'.format(estimated_var))
        print('Expected variance : {}'.format(expected_var))

    return estimated_exp,  estimated_var, expected_exp, expected_var

def influence_of_ro_over_estimation():
    '''
        Plot expectancy and variance error as a function of rho
    '''
    error_exp = []
    error_var = []
    ro_list = []
    for lbd in np.linspace(0.01, 0.99, 50):
        for mu in np.linspace(0.01, 0.99, 50):
            rho = lbd/mu
            if rho >= 1:
                continue
            est_exp, est_var, exp_exp, exp_var = estimate_exp_var(lbd=lbd, mu=mu)
            error_exp.append(abs(est_exp - exp_exp))
            error_var.append(abs(est_var - exp_var))
            ro_list.append(rho)

    plt.scatter(ro_list, error_exp, label='Expectancy error', s=20)
    plt.scatter(ro_list, error_var, label='Variance error', s=20)
    plt.xlabel('rho')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Influence of rho over estimation errors')
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
        s += duration * indicatrice(Xt[i], x)
    return s/t

def plot_density(T_max=10, N=10):
    '''
        Estimate Xt density by Monte Carlo
    '''
    X = np.arange(T_max)
    Y = np.zeros((N, T_max))
    for n in range(N):
        for x in X:
            Y[n, x] = density_Xt(x, t=T_max)
        print(n)
    Y_mean = np.mean(Y, axis=0)
    plt.plot(X, Y_mean, label='Estimated')
    plt.plot(X, compute_pi(n=T_max), label='Expected')
    plt.xlabel('Nombre de clients')
    plt.ylabel('Probabilité')
    plt.title('Densité du nombre de clients dans la file')
    plt.legend()
    plt.show()


def plot_dist_customer_n(n, N_max=N_max, lbd=lbd, mu=mu):
    '''
        Plot the distribution of Xn
    '''
    P = createP(N_max)
    Pn = np.linalg.matrix_power(P, n)
    dist_0 = np.zeros(N_max)
    dist_0[0] = 1.

    dist_n = np.dot(dist_0, Pn)

    pi = compute_pi(n=N_max, lbd=lbd, mu=mu)
    plt.plot(range(N_max), dist_n, label='Xn density')
    plt.plot(range(N_max), pi, label='pi density')
    plt.xlabel('Longueur de la file')
    plt.ylabel('Probabilité')
    plt.title('Distribution de la longueur de la file')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # QUESTION 1
    plot_Xt(T=1000, lbd=1, mu=2)
    plot_Xt(T=1000, lbd=2, mu=1)
    plot_Xt(T=1000, lbd=1, mu=1)
    
    # QUESTION 2
    plot_pi(50)
    print(estimate_exp_var(show=True))
    influence_of_ro_over_estimation()

    # QUESTION 3
    plot_density(T_max=30, N=300)

    # QUESTION 4
    plot_dist_customer_n(100, N_max=20)