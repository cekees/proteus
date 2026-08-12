#!/bin/bash

num_nodes=(4) #(1 2 4 8 16 32 64 84)
num_procs=(256) #(64 128 256 512 1024 2048 4096 5376) 
part_name=(workq) #(single workq workq workq workq workq workq workq) 
i=0
while [ $i -lt ${#num_nodes[*]} ]; do
    	
    # Replace the placeholders in the template with the current values"
    sed -e "s/NUM_NODES/${num_nodes[$i]}/g" -e "s/NUM_REFINEMENT/$(($i+2))/g" -e "s/QUEUE_NAME/${queue_name[$i]}/g" poisson.sh > poisson_slurm.sh 
    cat poisson_slurm.sh
    # Submit the Slurm script
    printf "Submitting job with nodes = ${num_nodes[$i]}" # and cores = ${num_procs[$i]} \n"
    
    sbatch poisson_slurm.sh

    rm poisson_slurm.sh

    # Optionally, add a delay to avoid overloading the cluster
    sleep 10

    i=$(( $i + 1));

done

