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
mpirun -np 36  ./sparc -name ././cyclix/MoS2_cyclix_SOC/temp_run/MoS2_cyclix_SOC -log_summary > ././cyclix/MoS2_cyclix_SOC/temp_run/MoS2_cyclix_SOC.log


mpirun -np 36  ./sparc -name ././ofdft/Al4_MD_OFDFT/temp_run/Al4_MD_OFDFT -log_summary > ././ofdft/Al4_MD_OFDFT/temp_run/Al4_MD_OFDFT.log


mpirun -np 36  ./sparc -name ././ofdft/Al4_non_orth_OFDFT/temp_run/Al4_non_orth_OFDFT -log_summary > ././ofdft/Al4_non_orth_OFDFT/temp_run/Al4_non_orth_OFDFT.log


