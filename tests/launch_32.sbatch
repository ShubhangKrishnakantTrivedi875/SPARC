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
mpirun -np 36  ./sparc -name ././highT/highT_O8/temp_run/highT_O8 -log_summary > ././highT/highT_O8/temp_run/highT_O8.log


mpirun -np 36  ./sparc -name ././highT/highT_Si8/temp_run/highT_Si8 -log_summary > ././highT/highT_Si8/temp_run/highT_Si8.log


mpirun -np 36  ./sparc -name ././highT/highT_C_PBE0/temp_run/highT_C_PBE0 -log_summary > ././highT/highT_C_PBE0/temp_run/highT_C_PBE0.log


