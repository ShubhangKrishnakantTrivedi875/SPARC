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
mpirun -np 200  ./sparc -name ././xc/mgga_tests/Si4_kpt_scan/temp_run/Si4_kpt_scan -log_summary > ././xc/mgga_tests/Si4_kpt_scan/temp_run/Si4_kpt_scan.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/Si4_kpt_rscan/temp_run/Si4_kpt_rscan -log_summary > ././xc/mgga_tests/Si4_kpt_rscan/temp_run/Si4_kpt_rscan.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/Si4_kpt_r2scan/temp_run/Si4_kpt_r2scan -log_summary > ././xc/mgga_tests/Si4_kpt_r2scan/temp_run/Si4_kpt_r2scan.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/Fe2_spin_scan_gamma/temp_run/Fe2_spin_scan_gamma -log_summary > ././xc/mgga_tests/Fe2_spin_scan_gamma/temp_run/Fe2_spin_scan_gamma.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/Fe2_spin_scan_kpt/temp_run/Fe2_spin_scan_kpt -log_summary > ././xc/mgga_tests/Fe2_spin_scan_kpt/temp_run/Fe2_spin_scan_kpt.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/Fe2_spin_rscan_kpt/temp_run/Fe2_spin_rscan_kpt -log_summary > ././xc/mgga_tests/Fe2_spin_rscan_kpt/temp_run/Fe2_spin_rscan_kpt.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/Fe2_spin_r2scan_kpt/temp_run/Fe2_spin_r2scan_kpt -log_summary > ././xc/mgga_tests/Fe2_spin_r2scan_kpt/temp_run/Fe2_spin_r2scan_kpt.log


mpirun -np 200  ./sparc -name ././cyclix/WSe2_cyclix/temp_run/WSe2_cyclix -log_summary > ././cyclix/WSe2_cyclix/temp_run/WSe2_cyclix.log


mpirun -np 200  ./sparc -name ././cyclix/HfSe2_cyclix/temp_run/HfSe2_cyclix -log_summary > ././cyclix/HfSe2_cyclix/temp_run/HfSe2_cyclix.log


mpirun -np 200  ./sparc -name ././cyclix/FeCl2_cyclix_spin/temp_run/FeCl2_cyclix_spin -log_summary > ././cyclix/FeCl2_cyclix_spin/temp_run/FeCl2_cyclix_spin.log


mpirun -np 200  ./sparc -name ././cyclix/NiCl2_cyclix_spin/temp_run/NiCl2_cyclix_spin -log_summary > ././cyclix/NiCl2_cyclix_spin/temp_run/NiCl2_cyclix_spin.log


mpirun -np 200  ./sparc -name ././cyclix/WS2_cyclix_SOC/temp_run/WS2_cyclix_SOC -log_summary > ././cyclix/WS2_cyclix_SOC/temp_run/WS2_cyclix_SOC.log


