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
mpirun -np 200  ./sparc -name ././Au_fcc211/temp_run/Au_fcc211 -log_summary > ././Au_fcc211/temp_run/Au_fcc211.log


mpirun -np 200  ./sparc -name ././Cu_FCC/temp_run/Cu_FCC -log_summary > ././Cu_FCC/temp_run/Cu_FCC.log


mpirun -np 200  ./sparc -name ././Mg_hcp/temp_run/Mg_hcp -log_summary > ././Mg_hcp/temp_run/Mg_hcp.log


mpirun -np 200  ./sparc -name ././MnAlCu2/temp_run/MnAlCu2 -log_summary > ././MnAlCu2/temp_run/MnAlCu2.log


mpirun -np 200  ./sparc -name ././MgO/temp_run1/MgO -log_summary > ././MgO/temp_run1/MgO.log


mpirun -np 200  ./sparc -name ././MgO/temp_run2/MgO -log_summary > ././MgO/temp_run2/MgO.log


mpirun -np 200  ./sparc -name ././MgO/temp_run3/MgO -log_summary > ././MgO/temp_run3/MgO.log


mpirun -np 200  ./sparc -name ././MoS2/temp_run1/MoS2 -log_summary > ././MoS2/temp_run1/MoS2.log


mpirun -np 200  ./sparc -name ././MoS2/temp_run2/MoS2 -log_summary > ././MoS2/temp_run2/MoS2.log


mpirun -np 200  ./sparc -name ././MoS2/temp_run3/MoS2 -log_summary > ././MoS2/temp_run3/MoS2.log


mpirun -np 200  ./sparc -name ././He16_NVKG/temp_run/He16_NVKG -log_summary > ././He16_NVKG/temp_run/He16_NVKG.log


mpirun -np 200  ./sparc -name ././He16_NVTNH/temp_run/He16_NVTNH -log_summary > ././He16_NVTNH/temp_run/He16_NVTNH.log


mpirun -np 200  ./sparc -name ././LiF_NVKG/temp_run/LiF_NVKG -log_summary > ././LiF_NVKG/temp_run/LiF_NVKG.log


mpirun -np 200  ./sparc -name ././O2_spin_spinparal_NVKG/temp_run/O2_spin_spinparal_NVKG -log_summary > ././O2_spin_spinparal_NVKG/temp_run/O2_spin_spinparal_NVKG.log


mpirun -np 200  ./sparc -name ././Si2_kpt_paral/temp_run/Si2_kpt_paral -log_summary > ././Si2_kpt_paral/temp_run/Si2_kpt_paral.log


mpirun -np 200  ./sparc -name ././Si2_domain_paral/temp_run/Si2_domain_paral -log_summary > ././Si2_domain_paral/temp_run/Si2_domain_paral.log


