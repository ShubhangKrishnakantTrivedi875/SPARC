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
mpirun -np 36  ./sparc -name ././Al18C2_NPTNP_onlyc/temp_run/Al18C2_NPTNP_onlyc -log_summary > ././Al18C2_NPTNP_onlyc/temp_run/Al18C2_NPTNP_onlyc.log


mpirun -np 36  ./sparc -name ././Al18C2_NPTNP_aeqb_ortho_c/temp_run/Al18C2_NPTNP_aeqb_ortho_c -log_summary > ././Al18C2_NPTNP_aeqb_ortho_c/temp_run/Al18C2_NPTNP_aeqb_ortho_c.log


mpirun -np 36  ./sparc -name ././Al18C2_NPTNP_ortho_c/temp_run/Al18C2_NPTNP_ortho_c -log_summary > ././Al18C2_NPTNP_ortho_c/temp_run/Al18C2_NPTNP_ortho_c.log


