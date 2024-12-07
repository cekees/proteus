#!/bin/bash

#SBATCH -N NUM_NODES
#SBATCH -n TASKS_PER_NODE
#SBATCH -t 20:00:00
#SBATCH -p PARTITION_NAME 
#SBATCH -A loni_ceds3d624
#SBATCH -J "poisson test"
#SBATCH -o poisson-%j.out	
#SBATCH -e poisson-%j.err
#SBATCH --mail-user=dnathawani@lsu.edu
#SBATCH --mail-type=END,FAIL

# below are job commands

echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Slurm Nodes Allocated          = $SLURM_JOB_NODELIST"
echo "Number of Nodes Allocated      = $SLURM_NNODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"

module purge
module load intel-mpi

# Set some handy environment variables.
export PROJECT_DIR=/project/$USER/proteus/test/ci
export WORK_DIR=/work/$USER/poisson-${SLURM_NNODES}_${SLURM_NTASKS}

export LD_LIBRARY_PATH=/project/darsh/miniforge/envs/petsc-dev/lib:$LD_LIBRARY_PATH

# mamba activate petsc-dev
#Make sure the WORK_DIR exists:
mkdir -p $WORK_DIR

# Copy files, jump to WORK_DIR, and execute a program
cp $PROJECT_DIR/poisson_3d_tetgen_p.py $WORK_DIR
cp $PROJECT_DIR/poisson_3d_tetgen_c0p1_n.py $WORK_DIR
# cp poisson.sh $WORK_DIR

cd $WORK_DIR

start_time=$(date +%s)
srun parun poisson_3d_tetgen_p.py poisson_3d_tetgen_c0p1_n.py -C "Refinement=NUM_REFINEMENT" -F -P "-ksp_rtol 1.0e-10 -ksp_type cg -pc_type bjacobi -sub_pc_type ilu -ksp_monitor" -p -l 2
end_time=$(date +%s)

# Mark the time it finishes.
echo "Date              = $(date)"
echo "Total time to run the job is $(($end_time - $start_time))"
echo $SLURM_NNODES  $SLURM_NTASKS  $(awk '/primitive calls/ {print $8}' $WORK_DIR/poisson_3d_tetgen_p.log | tail -n 1) >> ../poisson_time.txt
# exit the job
exit 0

