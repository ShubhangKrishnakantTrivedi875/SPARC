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
mpirun -np 200  ./sparc -name ././CuSi7/temp_run/CuSi7 -log_summary > ././CuSi7/temp_run/CuSi7.log


mpirun -np 200  ./sparc -name ././BaTiO3/temp_run/BaTiO3 -log_summary > ././BaTiO3/temp_run/BaTiO3.log


mpirun -np 200  ./sparc -name ././Fe_spin/temp_run/Fe_spin -log_summary > ././Fe_spin/temp_run/Fe_spin.log


mpirun -np 200  ./sparc -name ././H2O_sheet/temp_run1/H2O_sheet -log_summary > ././H2O_sheet/temp_run1/H2O_sheet.log


mpirun -np 200  ./sparc -name ././H2O_sheet/temp_run2/H2O_sheet -log_summary > ././H2O_sheet/temp_run2/H2O_sheet.log


mpirun -np 200  ./sparc -name ././H2O_sheet/temp_run3/H2O_sheet -log_summary > ././H2O_sheet/temp_run3/H2O_sheet.log


mpirun -np 200  ./sparc -name ././H2O_wire/temp_run1/H2O_wire -log_summary > ././H2O_wire/temp_run1/H2O_wire.log


mpirun -np 200  ./sparc -name ././H2O_wire/temp_run2/H2O_wire -log_summary > ././H2O_wire/temp_run2/H2O_wire.log


mpirun -np 200  ./sparc -name ././H2O_wire/temp_run3/H2O_wire -log_summary > ././H2O_wire/temp_run3/H2O_wire.log


mpirun -np 200  ./sparc -name ././O2_spin/temp_run/O2_spin -log_summary > ././O2_spin/temp_run/O2_spin.log


mpirun -np 200  ./sparc -name ././Si8_atom_geopt/temp_run/Si8_atom_geopt -log_summary > ././Si8_atom_geopt/temp_run/Si8_atom_geopt.log


mpirun -np 200  ./sparc -name ././Si8_cell_geopt/temp_run/Si8_cell_geopt -log_summary > ././Si8_cell_geopt/temp_run/Si8_cell_geopt.log


mpirun -np 200  ./sparc -name ././Si8_full_geopt/temp_run/Si8_full_geopt -log_summary > ././Si8_full_geopt/temp_run/Si8_full_geopt.log


mpirun -np 200  ./sparc -name ././Si8_kpt/temp_run/Si8_kpt -log_summary > ././Si8_kpt/temp_run/Si8_kpt.log


mpirun -np 200  ./sparc -name ././Si8/temp_run/Si8 -log_summary > ././Si8/temp_run/Si8.log


mpirun -np 200  ./sparc -name ././SiH4/temp_run/SiH4 -log_summary > ././SiH4/temp_run/SiH4.log


