#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 48:00:00
#SBATCH -p workq
#SBATCH -A loni_ceds3d624
#SBATCH -J CCS_TADR
#SBATCH -e CCS_TADR.err
#load proteus module and ensure proteus's python is in path
date
module purge
module load intel-mpi
#module purge
#module load proteus/fct
#export LD_LIBRARY_PATH=/home/packages/compilers/intel/compiler/2022.0.2/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}
export MV2_HOMOGENEOUS_CLUSTER=1
export LD_LIBRARY_PATH=/project/abarua4/miniforge/envs/petsc-dev/lib:$LD_LIBRARY_PATH
srun parun thelper_tadr_p.py thelper_tadr_n.py -l 5 -v -C "problem=1 STABILIZATION_TYPE=2 T=0.3 nDTout=200 refinement=2 FCT=False LUMPED_MASS_MATRIX=True physicalDiffusion=0.0"
#srun parun CCS_p.py CCS_n.py -l 5 -v -P "-ksp_type preonly -pc_type lu"
#srun parun CCS_p.py CCS_n.py -l 5 -v 

#srun parun --TwoPhaseFlow marin.py -F -l 5 -C "he=0.05"
exit 0

