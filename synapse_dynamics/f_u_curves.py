from lmfit import Minimizer, Parameters, report_fit
import numpy as np
import matplotlib.pylab as plt

x = np.arange(-2, 2, 0.01)

A = 0
K = 1
B = 5
Q = 0.01
mu = 0.1
T = 0.8
y = A + (K - A) / (1 + Q * np.exp(-B * (x - T))) ** (1 / mu)
plt.plot(x, y)

a = 1
b = 3
c = 7
y2 = a * np.exp(-np.exp(b - c * x))
plt.plot(x, y2)

a = 1
b = 5.29093166
c = 8.96366854
y3 = a * np.exp(-np.exp(b - c * x))
plt.plot(x, y3)

plt.legend(['Richard', 'Gompertz', 'Gompertz fitted'])

plt.axvline(x=0)
plt.axvline(x=1)

plt.show()
