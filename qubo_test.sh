#!/bin/bash

param_file="output/qubo_test/parameter.csv"
config_file="config.yaml"

while IFS=";" read -r n m def1 def2 def3 sweeps; do
    # Delete first row
    if [[ "$n" == "n" ]]; then
        continue
    fi

    default=($def1 $def2 $def3)

    for run in {0..2}; do
        # copy config file
        config_copy="output/qubo_test/config_${n}_${m}_shots${sweeps}_run${run}.yaml"
        cp "$config_file" "$config_copy"
        
        # compute hyperparameters
        one_indices="one_indices: ${default[run]}"
        def_number=$(echo "${default[run]}" | tr -d '[]' | tr ',' ' ' | wc -w)
        # mu_logic=$((n * m * 10))
        # mu_1c=$((n * m * 10))
        mu_logic=$(((n * m / 2) ** 2 / 2))        
        if (( mu_logic >= 15000 )); then
            mu_logic=15000
        fi
        mu_1c=$mu_logic
        mu_mon=$((def_number * 10))
        mu_conc=$((n / m * 10))
        mu_thr=$((mu_conc / 2))

        # Set in config.yaml the correct parameters
        sed -i "19s/.*/n_counterpart: $n/" "$config_copy"
        sed -i "20s/.*/grades: $m/" "$config_copy"
        sed -i "26s|.*|$one_indices|" "$config_copy"
        sed -i "60s/.*/    one_class: $mu_1c/" "$config_copy"
        sed -i "61s/.*/    logic: $mu_logic/" "$config_copy"
        sed -i "62s/.*/    monotonicity: $mu_mon/" "$config_copy"
        sed -i "63s/.*/    concentration: $mu_conc/" "$config_copy"
        sed -i "64s/.*/    min_thr: $mu_thr/" "$config_copy"
        sed -i "65s/.*/    max_thr: $mu_thr/" "$config_copy"
        sed -i "78s/.*/shots: $sweeps/" "$config_copy"

        echo "n: $n, m: $m, run $((run+1)), default=${default[run]}"
        python cost_function.py "$config_copy" > "output/qubo_test/test11_${n}_${m}_shots${sweeps}_run${run}.txt" 2>&1 &
        # python cost_function.py "$config_copy" > "output/qubo_test/test2_${n}_${m}_shots${shots}_run${run}.txt" 2>&1 &
    done
    wait

done < "$param_file"

rm -f output/qubo_test/config_*_run*.yaml