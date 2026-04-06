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
mpirun -np 36  ./sparc -name ././xc/exx_tests/NaCl_PBE0/temp_run/NaCl_PBE0 -log_summary > ././xc/exx_tests/NaCl_PBE0/temp_run/NaCl_PBE0.log


mpirun -np 36  ./sparc -name ././xc/exx_tests/O2_spin_HSE/temp_run/O2_spin_HSE -log_summary > ././xc/exx_tests/O2_spin_HSE/temp_run/O2_spin_HSE.log


mpirun -np 36  ./sparc -name ././PtAu_SOC/temp_run/PtAu_SOC -log_summary > ././PtAu_SOC/temp_run/PtAu_SOC.log


