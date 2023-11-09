import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

mean = 18.3
cv = 0.08
sigma = cv * mean

sample = np.random.normal(mean, sigma, 100000)
n, bins = np.histogram(sample, bins=100)

print(
    f'Mass in 1 sigma [{mean - sigma} | {mean} | {mean + sigma}]: {np.sum(n[np.where((bins > mean - sigma) & (bins < mean + sigma))]) / np.sum(n)} (68.27%)')
print(
    f'Mass in 2 sigma [{mean - 2 * sigma} | {mean} | {mean + 2 * sigma}]: {np.sum(n[np.where((bins > mean - 2 * sigma) & (bins < mean + 2 * sigma))]) / np.sum(n)} (95.45%)')
print(
    f'Mass in 3 sigma [{mean - 3 * sigma} | {mean} | {mean + 3 * sigma}]: {np.sum(n[np.where((bins > mean - 3 * sigma) & (bins < mean + 3 * sigma))]) / np.sum(n)} (99.73%)')

plt.hist(sample, bins=100)
plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
plt.axvline(mean + sigma, color='k', linestyle='dashed', linewidth=1)
plt.axvline(mean - sigma, color='k', linestyle='dashed', linewidth=1)
plt.axvline(mean + 2 * sigma, color='k', linestyle='dashed', linewidth=1)
plt.axvline(mean - 2 * sigma, color='k', linestyle='dashed', linewidth=1)
plt.axvline(mean + 3 * sigma, color='k', linestyle='dashed', linewidth=1)
plt.axvline(mean - 3 * sigma, color='k', linestyle='dashed', linewidth=1)
plt.show()
