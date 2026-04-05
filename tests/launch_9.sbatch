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
mpirun -np 200  ./sparc -name ././Fe3_noncollinear/temp_run/Fe3_noncollinear -log_summary > ././Fe3_noncollinear/temp_run/Fe3_noncollinear.log


mpirun -np 200  ./sparc -name ././FePt_noncollinear/temp_run/FePt_noncollinear -log_summary > ././FePt_noncollinear/temp_run/FePt_noncollinear.log


mpirun -np 200  ./sparc -name ././MnAu_noncollinear/temp_run/MnAu_noncollinear -log_summary > ././MnAu_noncollinear/temp_run/MnAu_noncollinear.log


mpirun -np 200  ./sparc -name ././mlff/AlSi_mlffflag1/temp_run/AlSi_mlffflag1 -log_summary > ././mlff/AlSi_mlffflag1/temp_run/AlSi_mlffflag1.log


mpirun -np 200  ./sparc -name ././mlff/AlSi_mlffflag21/temp_run/AlSi_mlffflag21 -log_summary > ././mlff/AlSi_mlffflag21/temp_run/AlSi_mlffflag21.log


mpirun -np 200  ./sparc -name ././mlff/AlSi_mlffflag22/temp_run/AlSi_mlffflag22 -log_summary > ././mlff/AlSi_mlffflag22/temp_run/AlSi_mlffflag22.log


mpirun -np 200  ./sparc -name ././mlff/C_sqmlff/temp_run/C_sqmlff -log_summary > ././mlff/C_sqmlff/temp_run/C_sqmlff.log


mpirun -np 200  ./sparc -name ././mlff/AlSi_mlff_nonorth/temp_run/AlSi_mlff_nonorth -log_summary > ././mlff/AlSi_mlff_nonorth/temp_run/AlSi_mlff_nonorth.log


mpirun -np 200  ./sparc -name ././mlff/HfSe2_cyclix_mlff/temp_run/HfSe2_cyclix_mlff -log_summary > ././mlff/HfSe2_cyclix_mlff/temp_run/HfSe2_cyclix_mlff.log


mpirun -np 200  ./sparc -name ././mlff/WSe2_cyclix_mlff/temp_run/WSe2_cyclix_mlff -log_summary > ././mlff/WSe2_cyclix_mlff/temp_run/WSe2_cyclix_mlff.log


mpirun -np 200  ./sparc -name ././MoO3_hubbard/temp_run/MoO3_hubbard -log_summary > ././MoO3_hubbard/temp_run/MoO3_hubbard.log


mpirun -np 200  ./sparc -name ././NiO_spin_kpts_non_ortho_hubbard/temp_run/NiO_spin_kpts_non_ortho_hubbard -log_summary > ././NiO_spin_kpts_non_ortho_hubbard/temp_run/NiO_spin_kpts_non_ortho_hubbard.log


