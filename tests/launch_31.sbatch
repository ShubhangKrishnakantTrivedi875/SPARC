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
mpirun -np 36  ./sparc -name ././highT/highT_B4C_MD/temp_run/highT_B4C_MD -log_summary > ././highT/highT_B4C_MD/temp_run/highT_B4C_MD.log


mpirun -np 36  ./sparc -name ././highT/highT_BN/temp_run/highT_BN -log_summary > ././highT/highT_BN/temp_run/highT_BN.log


mpirun -np 36  ./sparc -name ././highT/highT_H/temp_run/highT_H -log_summary > ././highT/highT_H/temp_run/highT_H.log


