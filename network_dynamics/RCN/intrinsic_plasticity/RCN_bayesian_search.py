import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll.base import Apply
from hyperopt.pyll.stochastic import sample

from helper_functions.recurrent_competitive_network import run_rcn

tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(tmp_folder)

from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials

import os


def cleanup(exit_code=None, frame=None):
    try:
        print("Cleaning up leftover files...")
        # delete tmp folder
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        pass

    # -- Kill all subprocesses
    mongod.terminate()
    for w in workers:
        w.terminate()


# -- folder and environment setup
atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []
for python_path in user_paths:
    if python_path.endswith('wm_colaboration'):
        break
print('PYTHONPATH:', python_path)
try:
    venv_path = os.environ['VIRTUAL_ENV']
except KeyError:
    try:
        venv_path = os.environ['CONDA_PREFIX']
    except KeyError:
        sys.exit('No virtual environment found')
print('VENV:', venv_path)

db_folder = f'{tmp_folder}/db'
os.mkdir(db_folder)
try:
    mongod = subprocess.Popen(['mongod', '--dbpath', db_folder, '--port', '1234'], stdout=subprocess.DEVNULL)
    print(f'Started mongod --dbpath {db_folder} --port 1234')
except:
    sys.exit('Could not start mongod')


# TODO the RCN is creating multiple subdirs in the tmp folder
def objective(x):
    r = run_rcn(x, tmp_folder=tmp_folder, plot=False, progressbar=False)

    return {'loss': -r['f1_score'], 'status': STATUS_OK, 'x': x,
            'ps_statistics': {'recall': r['recall'], 'accuracy': r['accuracy'], 'triggered': r['triggered'],
                              'spontaneous': r['spontaneous']}}


# Create the domain space (  hp.uniform(label, low, high) or hp.normal(label, mu, sigma)  )
param_dist = hp.uniform
# param_dist = hp.normal

space = {'background_activity': param_dist('background_activity', 10, 30),
         'i_e_weight': param_dist('i_e_weight', 5, 20),
         'e_i_weight': param_dist('e_i_weight', 0.5, 10),
         'e_e_max_weight': param_dist('e_e_weight', 5, 20),
         'i_frequency': param_dist('i_frequency', 10, 40),
         'cue_percentage': 100,
         'cue_length': 1,
         'num_attractors': 3,
         'attractor_size': 64,
         'network_size': 256,
         'num_cues': 10
         }

# Create the algorithm
tpe_algo = tpe.suggest

# Create trials objects ( MongoTrials is for parallel execution )
# trials = Trials()
trials = MongoTrials('mongo://localhost:1234/db/jobs')

workers = []
for i in range(os.cpu_count()):
    try:
        # -- The workers need to be started with the correct PYTHONPATH
        my_env = os.environ.copy()
        my_env["PYTHONPATH"] = "{}:{}".format(os.environ["PATH"], python_path)
        workers.append(subprocess.Popen(
            [f'{venv_path}/bin/hyperopt-mongo-worker', '--mongo', 'localhost:1234/db', '--workdir', tmp_folder],
            env=my_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ))
        print(
            f'{i} Started hyperopt-mongo-worker {venv_path}/bin/hyperopt-mongo-worker --mongo localhost:1234/db --workdir {tmp_folder}')
    except:
        sys.exit(f'Could not start hyperopt-mongo-worker {i}')

# Run the tpe algorithm
tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=trials,
                max_evals=2000, rstate=np.random.default_rng(50),
                max_queue_len=os.cpu_count())
results = trials.results
results = [r.to_dict() for r in results]

# -- Copy results into a dataframe
results_df = pd.DataFrame({
    'score': [x['loss'] for x in results],
    'params': [x['x'] for x in results],
    'iteration': list(range(len(results))),
    'recall': [x['ps_statistics']['recall'] for x in results],
    'accuracy': [x['ps_statistics']['accuracy'] for x in results],
    'triggered': [x['ps_statistics']['triggered'] for x in results],
    'spontaneous': [x['ps_statistics']['spontaneous'] for x in results]
})

# Sort with the highest score on top
results_df['score'] *= -1
results_df = results_df.sort_values('score', ascending=False)
results_df.to_csv(f'{tmp_folder}/results.csv')

# -- Kill all subprocesses
try:
    mongod.terminate()
    print('Terminated mongod')
except:
    print('Could not terminate mongod')
for i, w in enumerate(workers):
    try:
        w.terminate()
        print(f'Terminated hyperopt-mongo-worker {i}')
    except:
        print(f'Could not terminate hyperopt-mongo-worker {i}')

best_params = results_df.iloc[0]['params']
best_score = results_df.iloc[0]['score']
print(f'Best parameters: {best_params}')
best_string = '-sweep'
for k, v in best_params.items():
    if isinstance(space[k], Apply):
        best_string += f' {k}'
for k, v in best_params.items():
    if isinstance(space[k], Apply):
        best_string += f' -{k} {v}'
best_string += ' -joint_distribution -sigma 3 -num_samples 20 -cross_validation 3'
print('Use this string as command-line parameters for RCN_sweep.py:', best_string)

# Run model with the best parameters and plot output
run_rcn(best_params, plot=True, low_memory=False)

# TODO also save plot of best model
while True:
    save = input('Save results? (y/n)')
    if save == 'y':
        save_folder = f'SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
        os.makedirs(save_folder)
        # os.rename(f'{tmp_folder}/score.png', f'{save_folder}/score.png')
        os.rename(f'{tmp_folder}/results.csv', f'{save_folder}/results.csv')
        cleanup()
        break
    elif save == 'n':
        cleanup()
        break
    else:
        print("Please enter 'y' or 'n'")
