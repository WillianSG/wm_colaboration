#!/bin/bash
#
# exec_parallel_cluster.sh provides the ability to run multiple simulation at once, by launching individual jobs on the RUG peregrine cluster.
# if you want to run only one simulation e.g for plotting directly call your script.
#
# the command line arguments are <python executable path> <exec home path> <python file to run>
# dont use relative pathes
# the script will span as many jobs on the cluster as iterations definded below
#
# @author: o.j.richter@rug.nl
# adapted from https://github.com/rug-cit-hpc/cluster_course/blob/master/advanced_course/1.7_jobarray/solution/jobscript.sh

# standart for normal python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

## please adapt the following to your need:
# notify by email on fail and end
#SBATCH --mail-type=END,FAIL
# send email to
#SBATCH --mail-user=o.j.richter@rug.nl
# memory requirement - check on your PC first if possible
#SBATCH --mem=2GB
# max execution time
#SBATCH --time=00:29:00
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
    echo "ERROR: No valid python path defined $1"
    exit 1
fi

if [ -z "$2" ] || [ ! -d "$2" ]
then
    echo "ERROR: No valid dir $2"
    exit 1
fi

cd $2

if [ -z "$3" ] || [ ! -f "$3" ]
then
    echo "ERROR: No valid python file specified $3"
    pwd
    exit 1
fi

echo "run job $3 seed: ${SLURM_ARRAY_TASK_ID}"

$1 $3 ${SLURM_ARRAY_TASK_ID}
