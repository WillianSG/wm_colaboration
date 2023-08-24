#!/bin/bash
source /home/p291020/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate brian
PYTHONPATH=. python network_dynamics/RCN/intrinsic_plasticity/RCN_bayesian_search.py --n_evals 5000 --n_attractors 4
