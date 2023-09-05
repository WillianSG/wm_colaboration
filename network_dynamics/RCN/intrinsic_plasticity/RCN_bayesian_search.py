import argparse
import atexit
import multiprocessing
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
import socket

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, partial
from hyperopt.pyll.base import Apply
from hyperopt.pyll.stochastic import sample
from tqdm.auto import tqdm

from hyperopt import fmin, tpe, hp

from helper_functions.recurrent_competitive_network import run_rcn
from helper_functions.telegram_notify import TelegramNotify

parser = argparse.ArgumentParser()
parser.add_argument("--n_evals", type=int, default=2000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_cues", type=int, default=10)
parser.add_argument("--n_attractors", type=int, default=3)
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()

msg_args = ''
for k, v in vars(args).items():
    msg_args += f'{k}: {v}, '
telegram_bot = TelegramNotify()
main_msg_id = telegram_bot.send_timestamped_message(
    f'Starting Bayesian search with the following parameters: {msg_args}')

tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(tmp_folder)
print("TMP:", tmp_folder)

if args.parallel:
    from hyperopt.mongoexp import MongoTrials

import os


def cleanup(exit_code=None, frame=None):
    try:
        print("Cleaning up leftover tmp folder...")
        # delete tmp folder
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        print("No tmp folder to remove")
    except OSError:
        print("Cannot remove tmp folder")

    if args.parallel:
        # -- Kill all subprocesses
        mongod.terminate()
        for w in workers:
            w.terminate()


# -- folder and environment setup
atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

try:
    user_paths = os.environ["PYTHONPATH"].split(os.pathsep)
except KeyError:
    user_paths = []
for python_path in user_paths:
    if python_path.endswith("wm_colaboration"):
        break
print("PYTHONPATH:", python_path)
try:
    venv_path = os.environ["VIRTUAL_ENV"]
except KeyError:
    try:
        venv_path = os.environ["CONDA_PREFIX"]
    except KeyError:
        sys.exit("No virtual environment found")
print("VENV:", venv_path)


def objective(x):
    # r = run_rcn(x, tmp_folder=tmp_folder, progressbar=False, attractor_conflict_resolution='3')

    r = {'f1_score': 0.5, 'recall': 0.5, 'accuracy': 0.5, 'triggered': 0.5, 'spontaneous': 0.5}

    # create new instance because Bot is not pickleable
    telegram_bot = TelegramNotify()
    telegram_bot.edit_timestamped_message(
        f'*{1}/{args.n_evals}* Finished run.  Score: {r["f1_score"]}',
        telegram_msg_id)

    return {
        "loss": -r["f1_score"],
        "status": STATUS_OK,
        "x": x,
        "ps_statistics": {
            "recall": r["recall"],
            "accuracy": r["accuracy"],
            "triggered": r["triggered"],
            "spontaneous": r["spontaneous"],
        },
    }


# Create the domain space (  hp.uniform(label, low, high) or hp.normal(label, mu, sigma)  )
param_dist = hp.uniform
# param_dist = hp.normal

space = {
    "background_activity": param_dist("background_activity", 10, 30),
    "i_e_weight": param_dist("i_e_weight", 5, 20),
    "e_i_weight": param_dist("e_i_weight", 0.5, 10),
    "e_e_max_weight": param_dist("e_e_weight", 5, 20),
    "i_frequency": param_dist("i_frequency", 10, 40),
    "cue_percentage": 100,
    "cue_length": 1,
    "num_attractors": args.n_attractors,
    "attractor_size": 64,
    "network_size": 256,
    "num_cues": args.n_cues,
}

# Create the algorithm
tpe_algo = tpe.suggest

# Create Trials objects or MongoTrials for parallel execution
if args.parallel:
    sock = socket.socket()
    sock.bind(("localhost", 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    db_port = sock.getsockname()[1]
    sock.close()
    db_folder = f"{tmp_folder}/db"
    os.mkdir(db_folder)
    try:
        mongod = subprocess.Popen(
            ["mongod", "--dbpath", db_folder, "--port", f"{db_port}"],
            stdout=subprocess.DEVNULL,
        )
        print(f"Started mongod --dbpath {db_folder} --port {db_port}")
    except:
        sys.exit("Could not start mongod")

    trials = MongoTrials(f"mongo://localhost:{db_port}/db/jobs")
    workers = []
    for i in range(os.cpu_count() if args.n_workers == -1 else args.n_workers):
        try:
            # -- The workers need to be started with the correct PYTHONPATH
            my_env = os.environ.copy()
            my_env["PYTHONPATH"] = "{}:{}".format(os.environ["PATH"], python_path)
            workers.append(
                subprocess.Popen(
                    [
                        f"{venv_path}/bin/hyperopt-mongo-worker",
                        "--mongo",
                        f"localhost:{db_port}/db",
                        "--workdir",
                        tmp_folder,
                    ],
                    env=my_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            print(
                f"{i} Started hyperopt-mongo-worker {venv_path}/bin/hyperopt-mongo-worker --mongo localhost:{db_port}/db --workdir {tmp_folder}"
            )
        except:
            sys.exit(f"Could not start hyperopt-mongo-worker {i}")
else:
    trials = Trials()

telegram_msg_id = telegram_bot.reply_to_timestamped_message(f'*0/0* Waiting for first evaluation to finish.',
                                                            main_msg_id)

# Run the tpe algorithm
tpe_best = fmin(
    fn=objective,
    space=space,
    algo=tpe_algo,
    trials=trials,
    max_evals=args.n_evals,
    rstate=np.random.default_rng(50),
    max_queue_len=os.cpu_count(),
    show_progressbar=True,
)
results = trials.results
if args.parallel:
    results = [r.to_dict() for r in results]

# -- Copy results into a dataframe
results_df = pd.DataFrame(
    {
        "score": [x["loss"] for x in results],
        "params": [x["x"] for x in results],
        "iteration": list(range(len(results))),
        "recall": [x["ps_statistics"]["recall"] for x in results],
        "accuracy": [x["ps_statistics"]["accuracy"] for x in results],
        "triggered": [x["ps_statistics"]["triggered"] for x in results],
        "spontaneous": [x["ps_statistics"]["spontaneous"] for x in results],
    }
)

# Sort with the highest score on top
results_df["score"] *= -1
results_df = results_df.sort_values("score", ascending=False)
results_df.to_csv(f"{tmp_folder}/results.csv")

if args.parallel:
    # -- Kill all subprocesses
    try:
        mongod.terminate()
        print("Terminated mongod")
    except:
        print("Could not terminate mongod gracefully")
        subprocess.run(["killall", "mongod"])
    for i, w in enumerate(workers):
        try:
            w.terminate()
            print(f"Terminated hyperopt-mongo-worker {i}")
        except:
            print(f"Could not terminate hyperopt-mongo-worker {i} gracefully")
            subprocess.run(["killall", "hyperopt-mongo-worker"])

best_params = results_df.iloc[0]["params"]
best_score = results_df.iloc[0]["score"]

telegram_bot.reply_to_timestamped_message(
    f'Finished Bayesian search with the following results:\n{best_params}\nScore: {best_score}', main_msg_id)

# Save folder
save_folder = f'RESULTS/BAYESIAN_SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
os.makedirs(save_folder)

# Run model with the best parameters and plot output
# run_rcn(best_params, tmp_folder=tmp_folder, save_plot=save_folder, low_memory=False, attractor_conflict_resolution='3')
# os.rename(f'{tmp_folder}/score.png', f'{save_folder}/score.png')
os.rename(f"{tmp_folder}/results.csv", f"{save_folder}/results.csv")

print(f"Best parameters: {best_params}")
joint_parameter_string = "-sweep"
for k, v in best_params.items():
    if isinstance(space[k], Apply):
        joint_parameter_string += f" {k}"
for k, v in best_params.items():
    if isinstance(space[k], Apply):
        joint_parameter_string += f" -{k} {v}"
joint_parameter_string += " -joint_distribution -sigma 3 -num_samples 20 -cross_validation 3"
joint_parameter_string += f" -num_cues {args.n_cues} -num_attractors {args.n_attractors} -network_size 256 -attractor_size 64 -cue_length 1 -num_cues 1"
print("Use this string as command-line parameters for RCN_sweep.py:", joint_parameter_string)
with open(f"{save_folder}/string.txt", "w") as f:
    f.write(joint_parameter_string)

for par in best_params.items():
    if isinstance(space[par[0]], Apply):
        parameter_sweep_string = f"-sweep {par[0]}"
        for k, v in best_params.items():
            if isinstance(space[k], Apply):
                parameter_sweep_string += f" -{k} {v}"
        parameter_sweep_string += " -joint_distribution -sigma 3 -num_samples 20 -cross_validation 3"
        parameter_sweep_string += f" -num_cues {args.n_cues} -num_attractors {args.n_attractors} -network_size {args.n_attractors * (64 + 16)} -attractor_size {64} -cue_length 1"
        with open(f"{save_folder}/string.txt", "a") as f:
            f.write('\n' + parameter_sweep_string)

telegram_bot.reply_to_timestamped_message(f'Saved results to {save_folder}', main_msg_id)

cleanup()
