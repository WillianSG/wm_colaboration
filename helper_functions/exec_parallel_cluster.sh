#!/bin/bash
#
# exec_parallel_cluster.sh provides the ability to run multiple simulation at once, by launching individual jobs on the RUG peregrine cluster.
# if you want to run only one simulation e.g for plotting directly call your script.
#
# the command line arguments are <python executable path> <python file to run>
# the script will span as many jobs on the cluster as iterations definded below
#
# @author: o.j.richter@rug.nl
# adapted from https://github.com/rug-cit-hpc/cluster_course/blob/master/advanced_course/1.7_jobarray/solution/jobscript.sh

# standart for normal python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

## please adapt the following to your need:
# memory requirement - check on your PC first if possible
#SBATCH --mem=16GB
# max execution time
#SBATCH --time=01:00:00
# choose subcluster (short: up to 30 min, regular: up to 10 days - 128GB RAM?, himem: up tp 10 days - 1TB RAM?)
#SBATCH --partition=regular
# job name
#SBATCH --job-name=stdp_learning_rule
# from to (incl) range of iterations, max is 1000 total
#SBATCH --array=1-100

## Most importantly check your python script on pg-interactive.hpc.rug.nl first (with reduced load) if you environment works fine.

# Check if filename has been supplied
if [ -z "$1" ] || [ ! -x "$1" ]
then
    echo "ERROR: No valid python path defined"
    exit -1
fi

if [ -z "$2" ] || [ ! -f "$2" ]
then
    echo "ERROR: No valid python file specified"
    exit -1
fi

echo "run job $2 seed: ${SLURM_ARRAY_TASK_ID}"

$1 $2 ${SLURM_ARRAY_TASK_ID}
