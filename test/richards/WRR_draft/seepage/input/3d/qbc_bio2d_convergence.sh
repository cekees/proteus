#!/bin/bash
#SBATCH -N 10
#SBATCH -n 640 # 1920	
#SBATCH -t 72:00:00
#SBATCH -p workq
#SBATCH -A loni_ceds3d624
#SBATCH -J Bioswales
##SBATCH -e Infiltration.err
#SBATCH --mail-user abarua4@lsu.edu
#load proteus module and ensure proteus's python is in path
date
module purge
module load intel-mpi
export MV2_HOMOGENEOUS_CLUSTER=1
export LD_LIBRARY_PATH=/project/abarua4/miniforge/envs/petsc-dev/lib:$LD_LIBRARY_PATH

#export LD_LIBRARY_PATH=/home/packages/compilers/intel/compiler/2022.0.2/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}
srun parun Bioswales3d_p.py Bioswales3d_n.py -l 5 -v -P "-ksp_type preonly"
exit 0
