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
mpirun -np 36  ./sparc -name ././TiCrO4_kpt_hubbard/temp_run/TiCrO4_kpt_hubbard -log_summary > ././TiCrO4_kpt_hubbard/temp_run/TiCrO4_kpt_hubbard.log


mpirun -np 36  ./sparc -name ././Al_pr_relax_comp/temp_run/Al_pr_relax_comp -log_summary > ././Al_pr_relax_comp/temp_run/Al_pr_relax_comp.log


mpirun -np 36  ./sparc -name ././Al_pr_relax_tens/temp_run/Al_pr_relax_tens -log_summary > ././Al_pr_relax_tens/temp_run/Al_pr_relax_tens.log


