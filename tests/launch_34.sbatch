#!/bin/bash
#SBATCH -J SPARC_testsuite
#SBATCH -A gts-phanish6-coda20                    # Account name
#SBATCH -p inferno
#SBATCH -N 4 --ntasks-per-node=12
#SBATCH --mem-per-cpu=7G
#SBATCH -t1:00:00
cd $SLURM_SUBMIT_DIR

module purge
ml intel-oneapi-compilers intel-oneapi-mkl intel-oneapi-mpi

echo $PWD
mpirun -np 36  ./sparc -name ././mlff/AlSi_mlffflag1/temp_run/AlSi_mlffflag1 -log_summary > ././mlff/AlSi_mlffflag1/temp_run/AlSi_mlffflag1.log


mpirun -np 36  ./sparc -name ././mlff/AlSi_mlffflag21/temp_run/AlSi_mlffflag21 -log_summary > ././mlff/AlSi_mlffflag21/temp_run/AlSi_mlffflag21.log


mpirun -np 36  ./sparc -name ././mlff/AlSi_mlffflag22/temp_run/AlSi_mlffflag22 -log_summary > ././mlff/AlSi_mlffflag22/temp_run/AlSi_mlffflag22.log


