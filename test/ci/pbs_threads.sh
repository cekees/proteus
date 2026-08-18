#!/bin/bash

num_nodes=(1 2 4 8 16 32 64 128 256)
num_procs=(192 384 768 1536 3072 6144 12288 24576 49152)
part_name=(standard standard standard standard standard standard standard standard standard) 
i=0

while [ $i -lt ${#num_nodes[*]} ]; do
    	
    # Replace the placeholders in the template with the current values"
    sed -e "s/NUM_NODES/${num_nodes[$i]}/g" -e "s/TASKS_PER_NODE/${num_procs[$i]}/g" -e "s/NUM_REFINEMENT/$(($i+3))/g" -e "s/PARTITION_NAME/${part_name[$i]}/g" carpenter.pbs > poisson_submit.pbs 
        
    # Submit the Slurm script
    printf "Submitting job with nodes = ${num_nodes[$i]} and cores = ${num_procs[$i]} \n"
    
    qsub poisson_submit.pbs
    #cat poisson_submit.pbs
    
    rm poisson_submit.pbs

    # Optionally, add a delay to avoid overloading the cluster
    sleep 10

    i=$(( $i + 1));

done

