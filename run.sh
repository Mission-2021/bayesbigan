#!/bin/bash
#
#SBATCH --account=normal
#
#SBATCH --job-name=bigan
#
## output files
#SBATCH --output=exp/output/output-%j.log
#SBATCH --error=exp/output/output-%j.err
#
# Estimated running time. 
# The job will be killed when it runs 15 min longer than this time.
#SBATCH --time=2-0:00:00
#SBATCH --mem=70gb
#
## Resources 
## -p gpu/batch  |job type
## -N            |number of nodes
## -n            |number of cpu 
#SBATCH -p gpu 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclude=pgpu[01-03]

mkdir -p exp/output

source theanosetup.sh

#stdbuf -o0 ./train_check.sh
#stdbuf -o0 ./train_mnist.sh
stdbuf -o0 ./train_cifar.sh


#python check-gpu.py
