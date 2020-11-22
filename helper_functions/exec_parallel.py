"""
exec_parallel.py provides the ability to run multiple simulation at once, by using dedicated multithreading.
if you want to run only one simulation e.g for plotting directly call your script.

the command line arguments are <python file to run> <from> <to (excl)>
the script will span as many workers as there are CPU threads
the script will use the same pyhton executable as called with

@author: o.j.richter@rug.nl
"""
import threading
from queue import Queue
import subprocess
import sys
import multiprocessing
import logging

# get the python executable
PYTHON_COMMAND = sys.executable
if PYTHON_COMMAND is None:
    PYTHON_COMMAND = "python"
    logging.warning("python executable path not provided by the system using 'python'")

# adapt number of workers to system it is running on
MULTITHREAD_WORKER_THREADS = multiprocessing.cpu_count()

# read from to from command line
try:
    EXPERIMENT_FILE = sys.argv[1]
    MULTITHREAD_ITERATIONS_BEGIN = sys.argv[2]
    MULTITHREAD_ITERATIONS_END_BEFORE = sys.argv[3]
except IndexError:
    logging.error(f"could not read command line arguments: {sys.argv}")
    raise SystemExit(f"Usage: {sys.argv[0]} <file to run> <from> <to (excl)>")


def worker():
    while True:
        item = q.get()
        if item is None:
            break
        logging.info(f"started: {EXPERIMENT_FILE} seed: {item[0]}")
        subprocess.call([PYTHON_COMMAND, str(EXPERIMENT_FILE), str(item[0])])
        logging.info(f"finished: {EXPERIMENT_FILE} seed: {item[0]}")
        q.task_done()

#create workers
threads = []
for i in range(MULTITHREAD_WORKER_THREADS):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)
logging.info(f"{MULTITHREAD_WORKER_THREADS} workers created")

# create task queue
q = Queue()
for run_id in range(int(MULTITHREAD_ITERATIONS_BEGIN), int(MULTITHREAD_ITERATIONS_END_BEFORE)):
    item = [run_id]
    q.put(item)

# block until all tasks are done
q.join()

# stop workers
for i in range(MULTITHREAD_WORKER_THREADS):
    q.put(None)
for t in threads:
    t.join()