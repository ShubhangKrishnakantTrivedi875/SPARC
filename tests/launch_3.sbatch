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
mpirun -np 200  ./sparc -name ././TiNi_monoclinic/temp_run/TiNi_monoclinic -log_summary > ././TiNi_monoclinic/temp_run/TiNi_monoclinic.log


mpirun -np 200  ./sparc -name ././P_triclinic/temp_run/P_triclinic -log_summary > ././P_triclinic/temp_run/P_triclinic.log


mpirun -np 200  ./sparc -name ././BaTiO3_quick/temp_run/BaTiO3_quick -log_summary > ././BaTiO3_quick/temp_run/BaTiO3_quick.log


mpirun -np 200  ./sparc -name ././H2O_sheet_quick/temp_run/H2O_sheet_quick -log_summary > ././H2O_sheet_quick/temp_run/H2O_sheet_quick.log


mpirun -np 200  ./sparc -name ././H2O_wire_quick/temp_run/H2O_wire_quick -log_summary > ././H2O_wire_quick/temp_run/H2O_wire_quick.log


mpirun -np 200  ./sparc -name ././SiH4_quick/temp_run/SiH4_quick -log_summary > ././SiH4_quick/temp_run/SiH4_quick.log


mpirun -np 200  ./sparc -name ././Al18Si18_NPTNH/temp_run/Al18Si18_NPTNH -log_summary > ././Al18Si18_NPTNH/temp_run/Al18Si18_NPTNH.log


mpirun -np 200  ./sparc -name ././Al16Si16_NPTNH_restart/temp_run/Al16Si16_NPTNH_restart -log_summary > ././Al16Si16_NPTNH_restart/temp_run/Al16Si16_NPTNH_restart.log


mpirun -np 200  ./sparc -name ././Al18Si18_NPTNH_lat23/temp_run/Al18Si18_NPTNH_lat23 -log_summary > ././Al18Si18_NPTNH_lat23/temp_run/Al18Si18_NPTNH_lat23.log


mpirun -np 200  ./sparc -name ././Al18Si18_NPTNP/temp_run/Al18Si18_NPTNP -log_summary > ././Al18Si18_NPTNP/temp_run/Al18Si18_NPTNP.log


mpirun -np 200  ./sparc -name ././Al16Si16_NPTNP_restart/temp_run/Al16Si16_NPTNP_restart -log_summary > ././Al16Si16_NPTNP_restart/temp_run/Al16Si16_NPTNP_restart.log


mpirun -np 200  ./sparc -name ././Al18C2_NPTNP_aeqb_c/temp_run/Al18C2_NPTNP_aeqb_c -log_summary > ././Al18C2_NPTNP_aeqb_c/temp_run/Al18C2_NPTNP_aeqb_c.log


