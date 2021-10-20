""" Import libraries """
import numpy as np
import matplotlib.pyplot as plt

""" Define functions """
# Use trapezoid rule to integrate a function numerically from a to b
def integrate(X, Y):
    N = len(Y)
    deltax = (X[-1] - X[0])/N
    return (Y[0] + Y[N-1] + 2*np.sum(Y[1:N-2])) * deltax / 2

# Determine coefficients of Fourier series by numerical integration
def fourier(X, Y, N):
    a, b = [], []
    deltax = X[-1] - X[0]
    for n in range(N):
        a.append((2/deltax)*integrate(X, Y*np.cos(2*np.pi*n*X/deltax)))
        b.append((2/deltax)*integrate(X, Y*np.sin(2*np.pi*n*X/deltax)))
    return a, b

def f(x, a, b):
    period = x[-1] - x[0]
    func = a[0]/2 * np.ones_like(x)
    for i in range(1, len(a)):
        func += a[i] * np.cos(2*np.pi*i*x/period) + b[i] * np.sin(2*np.pi*i*x/period)
    return func

""" Main body """
# Import given wavedata from file and split into X and Y components
X = np.linspace(1, 10, 1000)
Y = 1/X

# Calculate Fourier coefficients
k = 20
a, b = fourier(X, Y, k)

# Input calculated a and b coefficients to obtain Fourier expansion of wave data
fourier = f(X, a, b)

""" Plotting and formatting """
fig = plt.figure(figsize = (15, 9))
ax3 = fig.add_subplot(111, title = f'Plot of Fourier reconstruction with terms up to n = {k}')
ax3.plot(X, Y, label = 'Plot of input data')
ax3.plot(X, fourier, label = f'Fourier approximation')
ax3.legend()

plt.show()