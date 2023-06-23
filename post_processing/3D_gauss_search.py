import numpy as np
import matplotlib.pyplot as plt
import pickle, sys, os

if sys.platform in ['linux', 'win32']:

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../../')))

sys.path.append(os.path.join(root, 'helper_functions'))

path = 'D://A_PhD//GitHub//wm_colaboration//results//sFSA_surface_gauss_search'

files = os.listdir(path)

dict_ = {}
params = []

for file in files:

    with open(
        f'{path}//{file}',
        'rb') as f:(
        pickle_dat) = pickle.load(f)

    _CR = 0

    for param, val in pickle_dat.items():

        if param not in dict_:
            dict_[param] = []

            if param != 'CR':
            	params.append(param)

        dict_[param].append(val)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(dict_['inh_rate'], dict_['w_acpt'], dict_['w_trans'], c=dict_['CR'], marker='s')

ax.set_xlabel('inh_rate')
ax.set_ylabel('w_acpt')
ax.set_zlabel('w_trans')

cbar = plt.colorbar(scatter)
cbar.set_label('CR')

plt.show()

dict_ = {}
params = []

for file in files:

    with open(
        f'{path}//{file}',
        'rb') as f:(
        pickle_dat) = pickle.load(f)

    _CR = 0

    for param, val in pickle_dat.items():
    	if param == 'CR':
    		_CR = val
    		break

    if _CR == 1.0:

	    for param, val in pickle_dat.items():

	        if param not in dict_:
	            dict_[param] = []

	            if param != 'CR':
	            	params.append(param)

	        dict_[param].append(val)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dict_['inh_rate'], dict_['w_acpt'], dict_['w_trans'], c='y', marker='o')

print(dict_['CR'])

ax.set_xlabel('inh_rate')
ax.set_ylabel('w_acpt')
ax.set_zlabel('w_trans')

plt.show()
