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
mpirun -np 200  ./sparc -name ././xc/vdW_tests/Si8_atom_geopt_d3/temp_run/Si8_atom_geopt_d3 -log_summary > ././xc/vdW_tests/Si8_atom_geopt_d3/temp_run/Si8_atom_geopt_d3.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/Si8_cell_geopt_d3/temp_run/Si8_cell_geopt_d3 -log_summary > ././xc/vdW_tests/Si8_cell_geopt_d3/temp_run/Si8_cell_geopt_d3.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/SnO_bulk_d3/temp_run/SnO_bulk_d3 -log_summary > ././xc/vdW_tests/SnO_bulk_d3/temp_run/SnO_bulk_d3.log


mpirun -np 200  ./sparc -name ././AlSi_orthogonal_quick_scf/temp_run/AlSi_orthogonal_quick_scf -log_summary > ././AlSi_orthogonal_quick_scf/temp_run/AlSi_orthogonal_quick_scf.log


mpirun -np 200  ./sparc -name ././AlSi_primitive_quick_relax/temp_run/AlSi_primitive_quick_relax -log_summary > ././AlSi_primitive_quick_relax/temp_run/AlSi_primitive_quick_relax.log


mpirun -np 200  ./sparc -name ././BN_primitive_quick_md/temp_run/BN_primitive_quick_md -log_summary > ././BN_primitive_quick_md/temp_run/BN_primitive_quick_md.log


mpirun -np 200  ./sparc -name ././LiNbO2_primitive_quick_scf/temp_run/LiNbO2_primitive_quick_scf -log_summary > ././LiNbO2_primitive_quick_scf/temp_run/LiNbO2_primitive_quick_scf.log


mpirun -np 200  ./sparc -name ././SiC_orthogonal_quick_relax/temp_run/SiC_orthogonal_quick_relax -log_summary > ././SiC_orthogonal_quick_relax/temp_run/SiC_orthogonal_quick_relax.log


mpirun -np 200  ./sparc -name ././TiO2_orthogonal_quick_md/temp_run/TiO2_orthogonal_quick_md -log_summary > ././TiO2_orthogonal_quick_md/temp_run/TiO2_orthogonal_quick_md.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/BaTiO3_scan/temp_run/BaTiO3_scan -log_summary > ././xc/mgga_tests/BaTiO3_scan/temp_run/BaTiO3_scan.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/BaTiO3_rscan/temp_run/BaTiO3_rscan -log_summary > ././xc/mgga_tests/BaTiO3_rscan/temp_run/BaTiO3_rscan.log


mpirun -np 200  ./sparc -name ././xc/mgga_tests/BaTiO3_r2scan/temp_run/BaTiO3_r2scan -log_summary > ././xc/mgga_tests/BaTiO3_r2scan/temp_run/BaTiO3_r2scan.log


