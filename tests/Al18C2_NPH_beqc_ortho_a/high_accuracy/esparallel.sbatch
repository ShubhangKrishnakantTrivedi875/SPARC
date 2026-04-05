#!/bin/bash
#SBATCH -J SPARC_testsuite
#SBATCH -A gts-phanish6
#SBATCH -qembers
#SBATCH -N3 --ntasks-per-node=20
#SBATCH --mem-per-cpu=2G
#SBATCH -o ./Reports/Report-%j.out
#SBATCH -t1:00:00
cd $SLURM_SUBMIT_DIR


module load intel-oneapi-compilers
module load intel-oneapi-mkl
module load intel-oneapi-mpi
#module load anaconda3
#conda activate krr_env

echo $PWD
#python SPARC_testing_script.py
#python SPARC_testing_script.py clean_temp
srun ../../../lib/sparc -name Al18C2_NPH_beqc_ortho_a -log_summary > Al18C2_NPH_beqc_ortho_a.log

