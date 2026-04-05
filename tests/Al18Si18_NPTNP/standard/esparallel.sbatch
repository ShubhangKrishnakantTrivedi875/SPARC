#!/bin/bash
#SBATCH -J SPARC_testsuite
#SBATCH -A gts-phanish6
#SBATCH -qembers
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=16G
#SBATCH -o ./Reports/Report-%j.out
#SBATCH -t2:00:00
cd $SLURM_SUBMIT_DIR


module load intel-oneapi-compilers
module load intel-oneapi-mkl
module load intel-oneapi-mpi

echo $PWD
mpirun ../../../lib/sparc -name Al18Si18_NPTNP -log_summary > Al18Si18_NPTNP.log

