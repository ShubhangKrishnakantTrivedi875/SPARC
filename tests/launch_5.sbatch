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
mpirun -np 200  ./sparc -name ././xc/vdW_tests/BaTiO3_vdWDF1/temp_run/BaTiO3_vdWDF1 -log_summary > ././xc/vdW_tests/BaTiO3_vdWDF1/temp_run/BaTiO3_vdWDF1.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/BaTiO3_vdWDF2/temp_run/BaTiO3_vdWDF2 -log_summary > ././xc/vdW_tests/BaTiO3_vdWDF2/temp_run/BaTiO3_vdWDF2.log


mpirun -np 200  ./sparc -name ././xc/exx_tests/C_HSE_aux/temp_run/C_HSE_aux -log_summary > ././xc/exx_tests/C_HSE_aux/temp_run/C_HSE_aux.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/Fe2_spin_gamma_ortho_vdWDF1/temp_run/Fe2_spin_gamma_ortho_vdWDF1 -log_summary > ././xc/vdW_tests/Fe2_spin_gamma_ortho_vdWDF1/temp_run/Fe2_spin_gamma_ortho_vdWDF1.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/Fe2_spin_kpt_nonOrtho_vdWDF2/temp_run/Fe2_spin_kpt_nonOrtho_vdWDF2 -log_summary > ././xc/vdW_tests/Fe2_spin_kpt_nonOrtho_vdWDF2/temp_run/Fe2_spin_kpt_nonOrtho_vdWDF2.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/MoS2_surface_d3/temp_run/MoS2_surface_d3 -log_summary > ././xc/vdW_tests/MoS2_surface_d3/temp_run/MoS2_surface_d3.log


mpirun -np 200  ./sparc -name ././xc/exx_tests/NaCl_PBE0/temp_run/NaCl_PBE0 -log_summary > ././xc/exx_tests/NaCl_PBE0/temp_run/NaCl_PBE0.log


mpirun -np 200  ./sparc -name ././xc/exx_tests/O2_spin_HSE/temp_run/O2_spin_HSE -log_summary > ././xc/exx_tests/O2_spin_HSE/temp_run/O2_spin_HSE.log


mpirun -np 200  ./sparc -name ././PtAu_SOC/temp_run/PtAu_SOC -log_summary > ././PtAu_SOC/temp_run/PtAu_SOC.log


mpirun -np 200  ./sparc -name ././xc/exx_tests/Si2_kpt_PBE0/temp_run/Si2_kpt_PBE0 -log_summary > ././xc/exx_tests/Si2_kpt_PBE0/temp_run/Si2_kpt_PBE0.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/Si4_kpt_vdWDF1/temp_run/Si4_kpt_vdWDF1 -log_summary > ././xc/vdW_tests/Si4_kpt_vdWDF1/temp_run/Si4_kpt_vdWDF1.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/Si4_kpt_vdWDF2/temp_run/Si4_kpt_vdWDF2 -log_summary > ././xc/vdW_tests/Si4_kpt_vdWDF2/temp_run/Si4_kpt_vdWDF2.log


