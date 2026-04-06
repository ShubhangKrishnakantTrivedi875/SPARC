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
#mpirun -np 36  ./sparc -name ././Al18Si18_NPTNP/temp_run/Al18Si18_NPTNP -log_summary > ././Al18Si18_NPTNP/temp_run/Al18Si18_NPTNP.log


mpirun -np 36  ./sparc -name ././Al16Si16_NPTNP_restart/temp_run/Al16Si16_NPTNP_restart -log_summary > ././Al16Si16_NPTNP_restart/temp_run/Al16Si16_NPTNP_restart.log


#mpirun -np 36  ./sparc -name ././Al18Si18_NPTNP_aeqb_c/temp_run/Al18Si18_NPTNP_aeqb_c -log_summary > ././Al18Si18_NPTNP_aeqb_c/temp_run/Al18Si18_NPTNP_aeqb_c.log


