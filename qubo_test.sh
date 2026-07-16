#!/bin/bash

param_file="output/gurobi_test/test1-parameters.csv"
config_file="config.yaml"

while IFS=";" read -r n m def1 def2 def3 sweeps; do
    # Delete first row
    if [[ "$n" == "n" ]]; then
        continue
    fi

    default=($def1 $def2 $def3)

    for run in {0..2}; do
        config_copy="output/gurobi_test/config_${n}_${m}_run${run}.yaml"
        cp "$config_file" "$config_copy"
        
        # compute hyperparameters
        one_indices="one_indices: ${default[run]}"
        def_number=$(echo "${default[run]}" | tr -d '[]' | tr ',' ' ' | wc -w)

        # Set in config.yaml the correct parameters
        sed -i "19s/.*/n_counterpart: $n/" "$config_copy"
        sed -i "20s/.*/grades: $m/" "$config_copy"
        sed -i "26s|.*|$one_indices|" "$config_copy"
        sed -i "86s/.*/shots: $sweeps/" "$config_copy"

        # run script
        echo "n: $n, m: $m, run $((run+1)), default=${default[run]}"
        python cost_function.py "$config_copy" > "output/gurobi_test/test01_${n}_${m}_run${run}.txt" 2>&1 &

    done
    wait

done < "$param_file"

rm -f output/gurobi_test/config_*_run*.yaml