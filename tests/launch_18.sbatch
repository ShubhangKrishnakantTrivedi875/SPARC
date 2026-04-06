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
mpirun -np 36  ./sparc -name ././xc/vdW_tests/Fe2_spin_gamma_ortho_vdWDF1/temp_run/Fe2_spin_gamma_ortho_vdWDF1 -log_summary > ././xc/vdW_tests/Fe2_spin_gamma_ortho_vdWDF1/temp_run/Fe2_spin_gamma_ortho_vdWDF1.log


mpirun -np 36  ./sparc -name ././xc/vdW_tests/Fe2_spin_kpt_nonOrtho_vdWDF2/temp_run/Fe2_spin_kpt_nonOrtho_vdWDF2 -log_summary > ././xc/vdW_tests/Fe2_spin_kpt_nonOrtho_vdWDF2/temp_run/Fe2_spin_kpt_nonOrtho_vdWDF2.log


mpirun -np 36  ./sparc -name ././xc/vdW_tests/MoS2_surface_d3/temp_run/MoS2_surface_d3 -log_summary > ././xc/vdW_tests/MoS2_surface_d3/temp_run/MoS2_surface_d3.log


