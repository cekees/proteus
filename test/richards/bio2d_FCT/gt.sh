#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:00:20
#SBATCH -p workq
#SBATCH -A loni_ceds3d
#SBATCH -J Bioswales
##SBATCH -e Infiltration.err
#SBATCH --mail-user abarua4@lsu.edu
#load proteus module and ensure proteus's python is in path
date
module purge
module load proteus/fct
#export LD_LIBRARY_PATH=/home/packages/compilers/intel/compiler/2022.0.2/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}
export MV2_HOMOGENEOUS_CLUSTER=1
mkdir -p $WORK/$SLURM_JOB_NAME.$SLURM_JOBID 
cd $SLURM_SUBMIT_DIR
cp qbc_bio2d_convergence.sh $WORK/$SLURM_JOB_NAME.$SLURM_JOBID

cp bio2d_n_coarse.poly $WORK/$SLURM_JOB_NAME.$SLURM_JOBID
cp bio2d_n_coarse.ele $WORK/$SLURM_JOB_NAME.$SLURM_JOBID
cp bio2d_n_coarse.node $WORK/$SLURM_JOB_NAME.$SLURM_JOBID

cp bio2d_convergence_p.py $WORK/$SLURM_JOB_NAME.$SLURM_JOBID
cp bio2d_convergence_n.py $WORK/$SLURM_JOB_NAME.$SLURM_JOBID
cp bio2d_convergence_n.xmf $WORK/$SLURM_JOB_NAME.$SLURM_JOBID
cp bio2d_convergence_n.h $WORK/$SLURM_JOB_NAME.$SLURM_JOBID
cd $WORK/$SLURM_JOB_NAME.$SLURM_JOBID
#srun parun --TwoPhaseFlow marin.py -F -l 5 -C "he=0.025"
#srun parun re_vgm_sand_10x10x10_3d_p.py re_vgm_sand_10x10x10_3d_c0p1_n.py -l 5 -v
#srun parun bio2d_convergence_p.py bio2d_convergence_n.py -l 5 -v -P "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type superlu_dist"
srun parun gatherTimes bio2d_convergence
#srun parun --TwoPhaseFlow marin.py -F -l 5 -C "he=0.05"
exit 0
