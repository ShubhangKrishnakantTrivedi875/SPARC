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
mpirun -np 36  ./sparc -name ././Al18Si18_NPTNH/temp_run/Al18Si18_NPTNH -log_summary > ././Al18Si18_NPTNH/temp_run/Al18Si18_NPTNH.log


mpirun -np 36  ./sparc -name ././Al16Si16_NPTNH_restart/temp_run/Al16Si16_NPTNH_restart -log_summary > ././Al16Si16_NPTNH_restart/temp_run/Al16Si16_NPTNH_restart.log


mpirun -np 36  ./sparc -name ././Al18Si18_NPTNH_lat23/temp_run/Al18Si18_NPTNH_lat23 -log_summary > ././Al18Si18_NPTNH_lat23/temp_run/Al18Si18_NPTNH_lat23.log


