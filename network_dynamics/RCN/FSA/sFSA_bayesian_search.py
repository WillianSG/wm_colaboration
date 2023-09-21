# -*- coding: utf-8 -*-
"""
@author: t.f.tiotto@rug.nl / w.soares.girao@rug.nl
@university: University of Groningen
@group: CogniGron
"""

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import socket

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll.base import Apply
from hyperopt.pyll.stochastic import sample


def cleanup(exit_code=None, frame=None):
    try:
        print("Cleaning up leftover tmp folder...")
        # delete tmp folder
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        print("No tmp folder to remove")
    except OSError:
        print("Cannot remove tmp folder")

    try:
        os.remove("logfile.txt")
    except FileNotFoundError:
        pass

    if args.parallel:
        # -- Kill all subprocesses
        mongod.terminate()
        for w in workers:
            w.terminate()


def is_pycharm():
    return os.getenv("PYCHARM_HOSTED") != None


if is_pycharm():
    from helper_functions.sFSA import run_sfsa
    from helper_functions.telegram_notify import TelegramNotify
else:
    root = os.path.dirname(os.path.abspath(os.path.join(__file__, "../")))
    sys.path.append(os.path.join(root, "helper_functions"))
    from sFSA import run_sfsa
    from telegram_notify import TelegramNotify

parser = argparse.ArgumentParser()
parser.add_argument("--n_evals", type=int, default=2000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()

telegram_token = "6491481149:AAFomgrhyBRohH4szH5jPT2_AoAdOYA_flY"
# telegram_token = '6488991500:AAEIZwY1f0dioEK-R8vPYMatnmmb_gCobZ8'  # Test

msg_args = ""
for k, v in vars(args).items():
    msg_args += f"{k}: {v}, "
telegram_bot = TelegramNotify(token=telegram_token)
telegram_bot.unpin_all()
main_msgs = telegram_bot.send_timestamped_messages(
    f"Starting FSA Bayesian search with the following parameters: {msg_args}"
)

tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(tmp_folder)
print("TMP:", tmp_folder)

if args.parallel:
    from hyperopt.mongoexp import MongoTrials

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
    r = run_sfsa(
        x,
        tmp_folder=tmp_folder,
        word_length=4,
        save_plot=False,
        already_in_tmp_folder=True if args.parallel else False,
    )

    # create new instance because Bot is not pickleable
    telegram_bot = TelegramNotify(token=telegram_token)
    # hack to keep track of last update
    telegram_bot.read_pinned_and_increment_it(telegram_msgs, r["f1_score"])

    return {
        "loss": -r["f1_score"],
        "status": STATUS_OK,
        "x": x,
        "ps_statistics": {"recall": r["recall"], "accuracy": r["accuracy"]},
    }


# Create the domain space (  hp.uniform(label, low, high) or hp.normal(label, mu, sigma)  )
param_dist = hp.uniform
# param_dist = hp.normal
# {'attractor_size': 64, 'background_activity': 13.855558143033827, 'cue_length': 1, 'cue_percentage': 100, 'e_e_max_weight': 14.337885250035841, 'e_i_weight': 4.253159333783978, 'i_e_weight': 5.302236785709991, 'i_frequency': 31.841289668071244, 'network_size': 256, 'num_attractors': 4, 'num_cues': 10}
space = {
    "background_activity": 13.855558143033827,
    "i_e_weight": 5.302236785709991,
    "e_i_weight": 4.253159333783978,
    "e_e_max_weight": 14.337885250035841,
    "i_frequency": 31.841289668071244,
    "cue_percentage": 100,
    "cue_length": param_dist("cue_length", 0.5, 1.0),
    "w_acpt": param_dist("w_acpt", 1.0, 4.0),
    "w_trans": param_dist("w_trans", 1.8, 13.8),
    "thr_GO_state": param_dist("thr_GO_state", -54.0, -42.0),
    "delay_A2GO": param_dist("delay_A2GO", 1.5, 3.0),
    "delay_gap_A2B": param_dist("delay_gap_A2B", 0.0, 0.3),
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

telegram_msgs = telegram_bot.reply_to_timestamped_messages(
    f"*0/{args.n_evals}* Waiting for first evaluation to finish.", main_msgs
)

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

telegram_bot.reply_to_timestamped_messages(
    f"Finished Bayesian search with the following results:\n{best_params}\nScore: {best_score}",
    main_msgs,
)

# Save folder
save_folder = (
    f'RESULTS/FSA_BAYESIAN_SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
)
os.makedirs(save_folder)

# Run model with the best parameters and plot output
run_sfsa(
    best_params,
    tmp_folder=tmp_folder,
    word_length=4,
    save_plot=True,
    seed_init=None,
    record_traces=True,
    save_path=save_folder,
)
os.rename(f"{tmp_folder}/results.csv", f"{save_folder}/results.csv")

telegram_bot.reply_to_timestamped_messages(f"Saved results to {save_folder}", main_msgs)
telegram_bot.unpin_all()

cleanup()
