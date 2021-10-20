""" Import libraries """
import numpy as np
import matplotlib.pyplot as plt

""" Define functions """
# Use trapezoid rule to integrate a function numerically from a to b
def integrate(X, Y):
    N = len(Y)
    dx = (X[-1] - X[0])/N
    result = 0

    result = Y[0]
    for k in range(1, N - 1):
        result += 2*Y[k - 1]
    result += Y[-1]

    return result * dx / 2

def fouriera(X, Y, N):
    a = []
    deltax = X[-1] - X[0]
    for n in range(N):
        a.append((2/deltax)*integrate(X, Y*np.cos(2*np.pi*n*X/deltax)))
    return a

def fourierb(X, Y, N):
    b = []
    deltax = X[-1] - X[0]
    for n in range(1, N):
        b.append((2/deltax)*integrate(X, Y*np.sin(2*np.pi*n*X/deltax)))
    return b

def fourier(X, Y, N):
    a = []
    b = []
    deltax = X[-1] - X[0]

    for n in range(N):
        a.append('')
        b.append('')
    return a, b

""" Main body """
wavedata = np.loadtxt("wavedata.csv", delimiter = ",")
X, Y = wavedata[:, 0], wavedata[:, 1]

print(integrate(X, Y))
print(fouriera(X, Y, 6))
print(fourierb(X, Y, 6))

""" Plotting and formatting """
fig = plt.figure(figsize = (12, 9))
plt.show()