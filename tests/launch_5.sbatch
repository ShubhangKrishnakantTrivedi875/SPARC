#!/bin/bash
#SBATCH -J SPARC_testsuite
#SBATCH -A gts-phanish6                    # Account name
#SBATCH -p embers
#SBATCH -N 4 --ntasks-per-node=12
#SBATCH --mem-per-cpu=7G
#SBATCH -t1:00:00
cd $SLURM_SUBMIT_DIR

module purge
ml intel-oneapi-compilers intel-oneapi-mkl intel-oneapi-mpi

echo $PWD
mpirun -np 48  ./sparc -name ././SiC_orthogonal_quick_relax/temp_run/SiC_orthogonal_quick_relax -log_summary > ././SiC_orthogonal_quick_relax/temp_run/SiC_orthogonal_quick_relax.log


