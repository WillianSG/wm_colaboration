import pickle, sys, os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import second, ms
import pickle
import pandas as pd
import matplotlib.colors as colors

if sys.platform in ['linux', 'win32']:

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../../')))

sys.path.append(os.path.join(root, 'helper_functions'))

path = 'D://A_PhD//GitHub//wm_colaboration//results//sFSA_surface_gauss_search'

files = os.listdir(path)

CR_ = []

for file in files:

    with open(
        f'{path}//{file}',
        'rb') as f:(
        pickle_dat) = pickle.load(f)

    _use = True
    cr = 0

    for param, val in pickle_dat.items():

    	if val < 0:
    		_use = False

    	if param == 'CR':
    		cr = val

    if _use:

    	CR_.append(val)

counts, bins, _ = plt.hist(CR_, bins = np.arange(0.0, 1.1, 0.1))

total_count = sum(counts)

# Add count percentage on top of each bar
for i in range(len(counts)):
    count = counts[i]
    bin_center = (bins[i] + bins[i+1]) / 2
    percentage = count / total_count * 100
    plt.text(bin_center, count, f'{percentage:.1f}%', ha='center', va='bottom', color = 'k')

plt.xlabel('sFSA CR')
plt.ylabel('count')
plt.ylim(0, 600)
plt.xlim(0, 1.0)
plt.xticks(np.arange(0.0, 1.1, 0.1))
plt.title(path.split('//')[-1])
plt.show()
plt.close()