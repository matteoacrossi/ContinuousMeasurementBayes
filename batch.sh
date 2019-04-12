#!/bin/bash
#SBATCH -A qtech
#SBATCH --job-name=fisher
#SBATCH --mail-user=matteo.rossi@unimi.it
#SBATCH --mail-type=END

export JULIA_PROJECT=ContinuousMeasurementBayes
julia -p $SLURM_CPUS_PER_TASK $1
