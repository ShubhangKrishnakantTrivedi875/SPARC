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
mpirun -np 36  ./sparc -name ././xc/vdW_tests/BaTiO3_vdWDF1/temp_run/BaTiO3_vdWDF1 -log_summary > ././xc/vdW_tests/BaTiO3_vdWDF1/temp_run/BaTiO3_vdWDF1.log


mpirun -np 36  ./sparc -name ././xc/vdW_tests/BaTiO3_vdWDF2/temp_run/BaTiO3_vdWDF2 -log_summary > ././xc/vdW_tests/BaTiO3_vdWDF2/temp_run/BaTiO3_vdWDF2.log


mpirun -np 36  ./sparc -name ././xc/exx_tests/C_HSE_aux/temp_run/C_HSE_aux -log_summary > ././xc/exx_tests/C_HSE_aux/temp_run/C_HSE_aux.log


