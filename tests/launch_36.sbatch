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
mpirun -np 36  ./sparc -name ././mlff/WSe2_cyclix_mlff/temp_run/WSe2_cyclix_mlff -log_summary > ././mlff/WSe2_cyclix_mlff/temp_run/WSe2_cyclix_mlff.log


mpirun -np 36  ./sparc -name ././MoO3_hubbard/temp_run/MoO3_hubbard -log_summary > ././MoO3_hubbard/temp_run/MoO3_hubbard.log


mpirun -np 36  ./sparc -name ././NiO_spin_kpts_non_ortho_hubbard/temp_run/NiO_spin_kpts_non_ortho_hubbard -log_summary > ././NiO_spin_kpts_non_ortho_hubbard/temp_run/NiO_spin_kpts_non_ortho_hubbard.log


