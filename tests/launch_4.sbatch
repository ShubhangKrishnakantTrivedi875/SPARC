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
mpirun -np 200  ./sparc -name ././Al18C2_NPTNP_onlyc/temp_run/Al18C2_NPTNP_onlyc -log_summary > ././Al18C2_NPTNP_onlyc/temp_run/Al18C2_NPTNP_onlyc.log


mpirun -np 200  ./sparc -name ././Al18C2_NPTNP_aeqb_ortho_c/temp_run/Al18C2_NPTNP_aeqb_ortho_c -log_summary > ././Al18C2_NPTNP_aeqb_ortho_c/temp_run/Al18C2_NPTNP_aeqb_ortho_c.log


mpirun -np 200  ./sparc -name ././Al18C2_NPTNP_ortho_c/temp_run/Al18C2_NPTNP_ortho_c -log_summary > ././Al18C2_NPTNP_ortho_c/temp_run/Al18C2_NPTNP_ortho_c.log


mpirun -np 200  ./sparc -name ././Al18C2_NPTNP_full_flex/temp_run/Al18C2_NPTNP_full_flex -log_summary > ././Al18C2_NPTNP_full_flex/temp_run/Al18C2_NPTNP_full_flex.log


mpirun -np 200  ./sparc -name ././Al18Si18_NPH/temp_run/Al18Si18_NPH -log_summary > ././Al18Si18_NPH/temp_run/Al18Si18_NPH.log


mpirun -np 200  ./sparc -name ././Al16Si16_NPH_restart/temp_run/Al16Si16_NPH_restart -log_summary > ././Al16Si16_NPH_restart/temp_run/Al16Si16_NPH_restart.log


mpirun -np 200  ./sparc -name ././Al18C2_NPH_aeqc_b/temp_run/Al18C2_NPH_aeqc_b -log_summary > ././Al18C2_NPH_aeqc_b/temp_run/Al18C2_NPH_aeqc_b.log


mpirun -np 200  ./sparc -name ././Al18C2_NPH_onlyb/temp_run/Al18C2_NPH_onlyb -log_summary > ././Al18C2_NPH_onlyb/temp_run/Al18C2_NPH_onlyb.log


mpirun -np 200  ./sparc -name ././Al18C2_NPH_beqc_ortho_a/temp_run/Al18C2_NPH_beqc_ortho_a -log_summary > ././Al18C2_NPH_beqc_ortho_a/temp_run/Al18C2_NPH_beqc_ortho_a.log


mpirun -np 200  ./sparc -name ././Al18C2_NPH_ortho_a/temp_run/Al18C2_NPH_ortho_a -log_summary > ././Al18C2_NPH_ortho_a/temp_run/Al18C2_NPH_ortho_a.log


mpirun -np 200  ./sparc -name ././Al18C2_NPH_full_flex/temp_run/Al18C2_NPH_full_flex -log_summary > ././Al18C2_NPH_full_flex/temp_run/Al18C2_NPH_full_flex.log


mpirun -np 200  ./sparc -name ././xc/vdW_tests/Au_wire_d3/temp_run/Au_wire_d3 -log_summary > ././xc/vdW_tests/Au_wire_d3/temp_run/Au_wire_d3.log


