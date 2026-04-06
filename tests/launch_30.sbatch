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
mpirun -np 36  ./sparc -name ././ofdft/Al4_Relax_OFDFT/temp_run/Al4_Relax_OFDFT -log_summary > ././ofdft/Al4_Relax_OFDFT/temp_run/Al4_Relax_OFDFT.log


mpirun -np 36  ./sparc -name ././ofdft/Al96_OFDFT/temp_run/Al96_OFDFT -log_summary > ././ofdft/Al96_OFDFT/temp_run/Al96_OFDFT.log


mpirun -np 36  ./sparc -name ././highT/highT_Al/temp_run/highT_Al -log_summary > ././highT/highT_Al/temp_run/highT_Al.log


