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
#mpirun -np 36  ./sparc -name ././Al18C2_NPTNP_full_flex/temp_run/Al18C2_NPTNP_full_flex -log_summary > ././Al18C2_NPTNP_full_flex/temp_run/Al18C2_NPTNP_full_flex.log


#mpirun -np 36  ./sparc -name ././Al18Si18_NPH/temp_run/Al18Si18_NPH -log_summary > ././Al18Si18_NPH/temp_run/Al18Si18_NPH.log


mpirun -np 36  ./sparc -name ././Al16Si16_NPH_restart/temp_run/Al16Si16_NPH_restart -log_summary > ././Al16Si16_NPH_restart/temp_run/Al16Si16_NPH_restart.log


