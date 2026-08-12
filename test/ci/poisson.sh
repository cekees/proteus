#!/bin/bash
#SBATCH -N NUM_NODES
#SBATCH -n TASKS_PER_NODE
#SBATCH -t 12:00:00
#SBATCH -p PARTITION_NAME 
#SBATCH -A hpc_ceds2d_2hp
#SBATCH -J "poisson test"
#SBATCH -o poisson-%j.out	
#SBATCH -e poisson-%j.err
#SBATCH --mail-user=cekees@lsu.edu
#SBATCH --mail-type=END,FAIL

# below are job commands
module swap cray-mpich cray-mpich-ucx
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Slurm Nodes Allocated          = $SLURM_JOB_NODELIST"
echo "Number of Nodes Allocated      = $SLURM_NNODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
eval "$(/p/home/cekees/miniforge3/bin/conda shell.bash hook)"

# Set some handy environment variables.
export PROJECT_DIR=/project/cekees/$USER/proteus_cekees/test/ci
export WORK_DIR=/work/$USER/poisson-${SLURM_NNODES}_${SLURM_NTASKS}_${SLURM_JOBID}

#Make sure the WORK_DIR exists:
mkdir -p $WORK_DIR

# Copy files, jump to WORK_DIR, and execute a program
cp poisson_slurm.sh $WORK_DIR
cp $PROJECT_DIR/poisson_3d_tetgen_p.py $WORK_DIR
cp $PROJECT_DIR/poisson_3d_tetgen_c0p1_n.py $WORK_DIR
# cp poisson.sh $WORK_DIR

cd $WORK_DIR
start_time=$(date +%s)
module list
which parun
srun parun poisson_3d_tetgen_p.py poisson_3d_tetgen_c0p1_n.py -C "Refinement=NUM_REFINEMENT" -F -P "-ksp_type cg -pc_type gamg" -m -M 4.0 -p -l 7
end_time=$(date +%s)

# Mark the time it finishes.
echo "Date              = $(date)"
echo "Total time to run the job is $(($end_time - $start_time))"
echo $SLURM_NNODES  $SLURM_NTASKS  $(awk '/Newton it/ {
    gsub(/\]/, "", $2);         # Clean the word
    if (val == "") {            # If this is the first match
        val = $2;               # Store it
    } else {                    # If we already have a stored value
        print $2 - val;         # Subtract current from stored and print
        val = "";               # Reset if you expect more pairs
    }
}' $WORK_DIR/poisson_3d_tetgen_p.log) >> ../poisson_time-hd2.txt 
#$(awk '/primitive calls/ {print $8}' $WORK_DIR/poisson_3d_tetgen_p.log | tail -n 1) >> ../poisson_time-hd2.txt
# exit the job
exit 0
