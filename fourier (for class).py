""" Import libraries """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
wavedata = np.loadtxt("wavedata.csv", delimiter = ",")
X, Y = wavedata[:, 0], wavedata[:, 1]

# Calculate Fourier coefficients up to 5 and up to 10
a5, b5 = fourier(X, Y, 5)
a10, b10 = fourier(X, Y, 10)

# Input calculated a and b coefficients to obtain Fourier expansion of wave data
fourier5 = f(X, a5, b5)
fourier10 = f(X, a10, b10)

""" Plotting and formatting """
fig = plt.figure(figsize = (15, 9))
gs = gridspec.GridSpec(ncols = 2, nrows = 2, figure = fig)

ax1 = fig.add_subplot(gs[0, 0], ylim = (min(a10 + b10) - 1, max(a10 + b10) + 1), xlabel = 'n', ylabel = 'Magnitude of corresponding cos term; a_n',
        title = 'Bar graph of coefficients of cos(2nπx/Period) up to n=10')
ax2 = fig.add_subplot(gs[0, 1], ylim = (min(a10 + b10) - 1, max(a10 + b10) + 1), xlabel = 'n', ylabel = 'Magnitude of corresponding sin term; b_n',
        title = 'Bar graph of coefficients of sin(2nπx/Period) up to n=10')
ax3 = fig.add_subplot(gs[1, 0], title = f'Plot of Fourier reconstruction with terms up to n = 5')
ax4 = fig.add_subplot(gs[1, 1], title = 'Plot of Fourier reconstruction with terms up to n = 10')

ax1.bar(np.arange(0, 10, 1), a10)
ax2.bar(np.arange(0, 10, 1), b10)

ax3.plot(X, Y, label = 'Plot of input data')
ax3.plot(X, fourier5, label = f'Fourier approximation')
ax3.legend()

ax4.plot(X, Y, label = 'Plot of input data')
ax4.plot(X, fourier10, label = 'Fourier approximation')
ax4.legend()

plt.show()