import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 1, 0.01)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, x)
ax2.plot(x, np.ones_like(x))
ax1.set_title('Functions')

a = 1
b = 5.29093166
c = 8.96366854
y2 = a * np.exp(-np.exp(b - c * x))
ax1.plot(x, y2)
f1_y2 = a * c * np.exp(b - np.exp(b - c * x) - c * x)
ax2.plot(x, f1_y2)

a = 1
b = 2.7083893552094156
c = 5.509734056519429
y3 = a * np.exp(-np.exp(b - c * x))
ax1.plot(x, y3)
f1_y3 = a * c * np.exp(b - np.exp(b - c * x) - c * x)
ax2.plot(x, f1_y3)
ax2.set_title('Derivatives')

ax1.legend(['Linear', 'Gompertz hand-fitted', 'Gompertz optimal'])
plt.suptitle('Gompertz vs linear')
plt.tight_layout()

plt.show()
