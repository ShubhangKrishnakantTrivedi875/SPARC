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
mpirun -np 200  ./sparc -name ././cyclix/MoS2_cyclix_SOC/temp_run/MoS2_cyclix_SOC -log_summary > ././cyclix/MoS2_cyclix_SOC/temp_run/MoS2_cyclix_SOC.log


mpirun -np 200  ./sparc -name ././ofdft/Al4_MD_OFDFT/temp_run/Al4_MD_OFDFT -log_summary > ././ofdft/Al4_MD_OFDFT/temp_run/Al4_MD_OFDFT.log


mpirun -np 200  ./sparc -name ././ofdft/Al4_non_orth_OFDFT/temp_run/Al4_non_orth_OFDFT -log_summary > ././ofdft/Al4_non_orth_OFDFT/temp_run/Al4_non_orth_OFDFT.log


mpirun -np 200  ./sparc -name ././ofdft/Al4_Relax_OFDFT/temp_run/Al4_Relax_OFDFT -log_summary > ././ofdft/Al4_Relax_OFDFT/temp_run/Al4_Relax_OFDFT.log


mpirun -np 200  ./sparc -name ././ofdft/Al96_OFDFT/temp_run/Al96_OFDFT -log_summary > ././ofdft/Al96_OFDFT/temp_run/Al96_OFDFT.log


mpirun -np 200  ./sparc -name ././highT/highT_Al/temp_run/highT_Al -log_summary > ././highT/highT_Al/temp_run/highT_Al.log


mpirun -np 200  ./sparc -name ././highT/highT_B4C_MD/temp_run/highT_B4C_MD -log_summary > ././highT/highT_B4C_MD/temp_run/highT_B4C_MD.log


mpirun -np 200  ./sparc -name ././highT/highT_BN/temp_run/highT_BN -log_summary > ././highT/highT_BN/temp_run/highT_BN.log


mpirun -np 200  ./sparc -name ././highT/highT_H/temp_run/highT_H -log_summary > ././highT/highT_H/temp_run/highT_H.log


mpirun -np 200  ./sparc -name ././highT/highT_O8/temp_run/highT_O8 -log_summary > ././highT/highT_O8/temp_run/highT_O8.log


mpirun -np 200  ./sparc -name ././highT/highT_Si8/temp_run/highT_Si8 -log_summary > ././highT/highT_Si8/temp_run/highT_Si8.log


mpirun -np 200  ./sparc -name ././highT/highT_C_PBE0/temp_run/highT_C_PBE0 -log_summary > ././highT/highT_C_PBE0/temp_run/highT_C_PBE0.log


