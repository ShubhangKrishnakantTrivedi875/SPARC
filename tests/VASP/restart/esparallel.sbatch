#!/bin/bash
#SBATCH -J SPARC_testsuite
#SBATCH -A gts-phanish6
#SBATCH -qembers
#SBATCH -N3 --ntasks-per-node=20
#SBATCH --mem-per-cpu=2G
#SBATCH -o ./Reports/Report-%j.out
#SBATCH -t8:00:00
cd $SLURM_SUBMIT_DIR


module load intel-oneapi-compilers
module load intel-oneapi-mkl
module load intel-oneapi-mpi

echo $PWD
mpirun ../../../lib/sparc -name vasp_al_c -log_summary > vasp_al_c3.log



