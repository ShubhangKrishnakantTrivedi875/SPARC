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
mpirun -np 36  ./sparc -name ././xc/mgga_tests/BaTiO3_scan/temp_run/BaTiO3_scan -log_summary > ././xc/mgga_tests/BaTiO3_scan/temp_run/BaTiO3_scan.log


mpirun -np 36  ./sparc -name ././xc/mgga_tests/BaTiO3_rscan/temp_run/BaTiO3_rscan -log_summary > ././xc/mgga_tests/BaTiO3_rscan/temp_run/BaTiO3_rscan.log


mpirun -np 36  ./sparc -name ././xc/mgga_tests/BaTiO3_r2scan/temp_run/BaTiO3_r2scan -log_summary > ././xc/mgga_tests/BaTiO3_r2scan/temp_run/BaTiO3_r2scan.log


