#!/bin/bash

param_file="output/qubo_test/parameter.csv"
config_file="config.yaml"

while IFS=";" read -r n m def1 def2 def3 mu_onec mu_logic mu_mon mu_conc mu_thr shots; do
    # Delete first row
    if [[ "$n" == "n" ]]; then
        continue
    fi

    # Set in config.yaml the correct parameters
    sed -i "19s/.*/n_counterpart: $n/" "$config_file"
    sed -i "20s/.*/grades: $m/" "$config_file"
    sed -i "60s/.*/    one_class: $mu_onec/" "$config_file"
    sed -i "61s/.*/    logic: $mu_logic/" "$config_file"
    sed -i "62s/.*/    monotonicity: $mu_mon/" "$config_file"
    sed -i "63s/.*/    concentration: $mu_conc/" "$config_file"
    sed -i "64s/.*/    min_thr: $mu_thr/" "$config_file"
    sed -i "65s/.*/    max_thr: $mu_thr/" "$config_file"
    sed -i "78s/.*/shots: $shots/" "$config_file"

    default=($def1 $def2 $def3)

    for run in {0..2}; do
        config_copy="output/qubo_test/config_${n}_${m}_shots${shots}_run${run}.yaml"
        cp "$config_file" "$config_copy"
        one_indices="one_indices: ${default[run]}"
        sed -i "26s|.*|$one_indices|" "$config_copy"

        echo "n: $n, m: $m, run $((run+1)), default=${default[run]}"
        python cost_function.py "$config_copy" > "output/qubo_test/output_${n}_${m}_shots${shots}_run${run}.txt" 2>&1 &
    done
    wait

done < "$param_file"

rm -f output/qubo_test/config_*_run*.yaml