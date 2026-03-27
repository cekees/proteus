#!/bin/bash

#num_nodes=(1 2 4 8 16 32 64 128)
#queue_name=(standard standard standard standard standard standard standard standard) 
num_nodes=(1 2) # 4 8 16 32 64 84)
queue_name=(debug debug) # standard standard standard standard standard standard) 
#num_nodes=(1 2 4 8)
#queue_name=(debug debug debug debug)
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

