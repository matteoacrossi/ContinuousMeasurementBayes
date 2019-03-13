#!/bin/bash
#SBATCH -A qtech
#SBATCH -c 6
#SBATCH --job-name=omega
#SBATCH --mail-user=matteo.rossi@unimi.it
#SBATCH --mail-type=END

srun --unbuffered julia --project=. -p $SLURM_CPUS_PER_TASK test_various_omegas_dt001.jl
mv Estimation_vs_omega.png Estimation_vs_omega_dt001.png
