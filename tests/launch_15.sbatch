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
mpirun -np 36  ./sparc -name ././Al18Si18_NPH_aeqc_b/temp_run/Al18Si18_NPH_aeqc_b -log_summary > ././Al18Si18_NPH_aeqc_b/temp_run/Al18Si18_NPH_aeqc_b.log


mpirun -np 36  ./sparc -name ././Al18C2_NPH_onlyb/temp_run/Al18C2_NPH_onlyb -log_summary > ././Al18C2_NPH_onlyb/temp_run/Al18C2_NPH_onlyb.log


mpirun -np 36  ./sparc -name ././Al18C2_NPH_beqc_ortho_a/temp_run/Al18C2_NPH_beqc_ortho_a -log_summary > ././Al18C2_NPH_beqc_ortho_a/temp_run/Al18C2_NPH_beqc_ortho_a.log


