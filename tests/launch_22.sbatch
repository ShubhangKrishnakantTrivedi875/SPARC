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
mpirun -np 36  ./sparc -name ././AlSi_orthogonal_quick_scf/temp_run/AlSi_orthogonal_quick_scf -log_summary > ././AlSi_orthogonal_quick_scf/temp_run/AlSi_orthogonal_quick_scf.log


mpirun -np 36  ./sparc -name ././AlSi_primitive_quick_relax/temp_run/AlSi_primitive_quick_relax -log_summary > ././AlSi_primitive_quick_relax/temp_run/AlSi_primitive_quick_relax.log


mpirun -np 36  ./sparc -name ././BN_primitive_quick_md/temp_run/BN_primitive_quick_md -log_summary > ././BN_primitive_quick_md/temp_run/BN_primitive_quick_md.log


