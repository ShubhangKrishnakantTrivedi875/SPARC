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
mpirun -np 36  ./sparc -name ././mlff/C_sqmlff/temp_run/C_sqmlff -log_summary > ././mlff/C_sqmlff/temp_run/C_sqmlff.log


mpirun -np 36  ./sparc -name ././mlff/AlSi_mlff_nonorth/temp_run/AlSi_mlff_nonorth -log_summary > ././mlff/AlSi_mlff_nonorth/temp_run/AlSi_mlff_nonorth.log


mpirun -np 36  ./sparc -name ././mlff/HfSe2_cyclix_mlff/temp_run/HfSe2_cyclix_mlff -log_summary > ././mlff/HfSe2_cyclix_mlff/temp_run/HfSe2_cyclix_mlff.log


