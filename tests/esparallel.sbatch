#!/bin/bash
#SBATCH -J SPARC_testsuite
#SBATCH -A gts-phanish6
#SBATCH -qembers
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=1G
#SBATCH -o ./Reports/Report-%j.out
#SBATCH -t2:00:00
cd $SLURM_SUBMIT_DIR


module load intel-oneapi-compilers
module load intel-oneapi-mkl
module load intel-oneapi-mpi
module load anaconda3
conda activate krr_env

echo $PWD
python SPARC_testing_script.py -systems Al16Si16_NPTNP_restart Al16Si16_NPH_restart
#python SPARC_testing_script.py clean_temp
#mpirun ../../../lib/sparc -name vasp_al_c -log_summary > vasp_al_c3.log


