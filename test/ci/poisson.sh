#!/bin/bash
#SBATCH -N NUM_NODES
#SBATCH --ntasks-per-node 128
#SBATCH -t 00:30:00
#SBATCH -q QUEUE_NAME 
#SBATCH -A ARONC51302008
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

conda activate petsc-dev

export PROTEUS_ENV=/p/home/cekees/miniforge3/envs/petsc-dev

export LD_LIBRARY_PATH="$PROTEUS_ENV/lib:$LD_LIBRARY_PATH"

export WORK_DIR=$WORKDIR/poisson-hd2-${SLURM_NNODES}_${SLURM_NTASKS}
export MESH_DIR=$WORKDIR/poisson-${SLURM_NNODES}_${SLURM_NTASKS}

#Make sure the WORK_DIR exists:
mkdir -p $WORK_DIR

# Copy files, jump to WORK_DIR, and execute a program
cp $SLURM_SUBMIT_DIR/poisson_3d_tetgen_p.py $WORK_DIR
cp $SLURM_SUBMIT_DIR/poisson_3d_tetgen_c0p1_n.py $WORK_DIR
cp $SLURM_SUBMIT_DIR/poisson.sh $WORK_DIR
#cp $MESH_DIR/meshNoVessel.* $WORK_DIR
cd $WORK_DIR
start_time=$(date +%s)
export UCX_UD_TIMEOUT=2m
srun parun poisson_3d_tetgen_p.py poisson_3d_tetgen_c0p1_n.py -C "Refinement=NUM_REFINEMENT genMesh=True" -F -P "-ksp_rtol 0.0 -ksp_atol 1.0e-9 -ksp_type cg -pc_type gamg -log_view" -l 5 -m
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
