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
mpirun -np 36  ./sparc -name ././cyclix/FeCl2_cyclix_spin/temp_run/FeCl2_cyclix_spin -log_summary > ././cyclix/FeCl2_cyclix_spin/temp_run/FeCl2_cyclix_spin.log


mpirun -np 36  ./sparc -name ././cyclix/NiCl2_cyclix_spin/temp_run/NiCl2_cyclix_spin -log_summary > ././cyclix/NiCl2_cyclix_spin/temp_run/NiCl2_cyclix_spin.log


mpirun -np 36  ./sparc -name ././cyclix/WS2_cyclix_SOC/temp_run/WS2_cyclix_SOC -log_summary > ././cyclix/WS2_cyclix_SOC/temp_run/WS2_cyclix_SOC.log


