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
mpirun -np 36  ./sparc -name ././Fe3_noncollinear/temp_run/Fe3_noncollinear -log_summary > ././Fe3_noncollinear/temp_run/Fe3_noncollinear.log


mpirun -np 36  ./sparc -name ././FePt_noncollinear/temp_run/FePt_noncollinear -log_summary > ././FePt_noncollinear/temp_run/FePt_noncollinear.log


mpirun -np 36  ./sparc -name ././MnAu_noncollinear/temp_run/MnAu_noncollinear -log_summary > ././MnAu_noncollinear/temp_run/MnAu_noncollinear.log


