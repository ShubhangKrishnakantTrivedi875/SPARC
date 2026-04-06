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
mpirun -np 36  ./sparc -name ././xc/mgga_tests/Fe2_spin_r2scan_kpt/temp_run/Fe2_spin_r2scan_kpt -log_summary > ././xc/mgga_tests/Fe2_spin_r2scan_kpt/temp_run/Fe2_spin_r2scan_kpt.log


mpirun -np 36  ./sparc -name ././cyclix/WSe2_cyclix/temp_run/WSe2_cyclix -log_summary > ././cyclix/WSe2_cyclix/temp_run/WSe2_cyclix.log


mpirun -np 36  ./sparc -name ././cyclix/HfSe2_cyclix/temp_run/HfSe2_cyclix -log_summary > ././cyclix/HfSe2_cyclix/temp_run/HfSe2_cyclix.log


