""" Import libraries """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def fourier(X, Y, N):
    a = []
    b = []
    deltax = X[-1] - X[0]

    for n in range(N):
        a.append((2/deltax)*integrate(X, Y*np.cos(2*np.pi*n*X/deltax)))
        b.append((2/deltax)*integrate(X, Y*np.sin(2*np.pi*n*X/deltax)))
    return a, b

def f(x, a, b):
    P = x[-1] - x[0]
    func = np.zeros_like(x)
    func += a[0]/2
    for i in range(1, len(a)):
        func += a[i] * np.cos(2*np.pi*i*x/P) + b[i] * np.sin(2*np.pi*i*x/P)
    return func

""" Main body """
wavedata = np.loadtxt("wavedata.csv", delimiter = ",")
X, Y = wavedata[:, 0], wavedata[:, 1]

iterations = 6
a, b = fourier(X, Y, iterations)
a10, b10 = fourier(X, Y, 10)

fouriery = f(X, a, b)
fouriery10 = f(X, a10, b10)

""" Plotting and formatting """
fig = plt.figure(figsize = (15, 9))
gs = gridspec.GridSpec(ncols = 2, nrows = 2, figure = fig)
ax1 = fig.add_subplot(gs[0, 0], ylim = (min(a10 + b10) - 1, max(a10 + b10) + 1), xlabel = 'n', ylabel = 'Magnitude of corresponding cos term')
ax2 = fig.add_subplot(gs[0, 1], ylim = (min(a10 + b10) - 1, max(a10 + b10) + 1), xlabel = 'n', ylabel = 'Magnitude of corresponding sin term')
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1], title = 'Hi')

range = np.arange(0, 10, 1)
ax1.bar(range, a10)
ax2.bar(range, b10)

ax3.plot(X, Y, label = 'Plot of input data')
ax3.plot(X, fouriery, label = 'Plot of Fourier approximation with sin and cos terms up to n=6')
ax3.legend()

ax4.plot(X, Y, label = 'Plot of input data')
ax4.plot(X, fouriery10, label = 'Plot of Fourier approximation with sin and cos terms up to n=10')
ax4.legend()

plt.show()