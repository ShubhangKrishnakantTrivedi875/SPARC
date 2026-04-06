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
mpirun -np 36  ./sparc -name ././xc/mgga_tests/Fe2_spin_scan_gamma/temp_run/Fe2_spin_scan_gamma -log_summary > ././xc/mgga_tests/Fe2_spin_scan_gamma/temp_run/Fe2_spin_scan_gamma.log


mpirun -np 36  ./sparc -name ././xc/mgga_tests/Fe2_spin_scan_kpt/temp_run/Fe2_spin_scan_kpt -log_summary > ././xc/mgga_tests/Fe2_spin_scan_kpt/temp_run/Fe2_spin_scan_kpt.log


mpirun -np 36  ./sparc -name ././xc/mgga_tests/Fe2_spin_rscan_kpt/temp_run/Fe2_spin_rscan_kpt -log_summary > ././xc/mgga_tests/Fe2_spin_rscan_kpt/temp_run/Fe2_spin_rscan_kpt.log


