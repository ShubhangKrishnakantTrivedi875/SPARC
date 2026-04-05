#!/bin/bash
#SBATCH -J SPARC_testsuite
#SBATCH -A gts-phanish6
#SBATCH -qembers
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH -o ./Reports/Report-%j.out
#SBATCH -t0:20:00
cd $SLURM_SUBMIT_DIR


module load intel-oneapi-compilers
module load intel-oneapi-mkl
module load intel-oneapi-mpi

echo $PWD
mpirun ../../../lib/sparc -name AlSi_orthogonal_quick_scf -log_summary > AlSi_orthogonal_quick_scf.log

~                                                                                                                               
