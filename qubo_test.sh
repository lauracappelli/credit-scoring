#!/bin/bash

param_file="output/annealing/parameters.csv"
config_file="config.yaml"

while IFS=";" read -r n m def1 def2 def3 sweeps ann_type mu_type mu1 mu2 mu3 mu4 mu5 mu6 mu7; do
    # Delete first row
    if [[ "$n" == "n" ]]; then
        continue
    fi

    default=($def1 $def2 $def3)

    output_folder="test"
    file_tag="test"
    if [[ "$ann_type" == "quantum" ]]; then
        output_folder="quantum"
        file_tag="qsa"
    elif [[ "$ann_type" == "classical" ]]; then
        output_folder="classical"
        file_tag="sa"
    fi

    for run in {0..2}; do
        config_copy="output/annealing/${output_folder}/config_${n}_${m}_run${run}.yaml"
        cp "$config_file" "$config_copy"
        
        # compute hyperparameters
        one_indices="one_indices: ${default[run]}"
        def_number=$(echo "${default[run]}" | tr -d '[]' | tr ',' ' ' | wc -w)

        # Set in config.yaml the correct parameters
        sed -i "19s/.*/n_counterpart: $n/" "$config_copy"
        sed -i "20s/.*/grades: $m/" "$config_copy"
        sed -i "26s|.*|$one_indices|" "$config_copy"
        sed -i "86s/.*/shots: $sweeps/" "$config_copy"

        if [[ "$ann_type" == "quantum" ]]; then
            sed -i "42s/.*/    annealing: False/" "$config_copy"
            sed -i "44s/.*/    quantum_annealing: True/" "$config_copy"
        elif [[ "$ann_type" == "classical" ]]; then
            sed -i "42s/.*/    annealing: True/" "$config_copy"
            sed -i "44s/.*/    quantum_annealing: False/" "$config_copy"
        fi

        sed -i "60s/.*/mu_table: $mu_type/" "$config_copy"
        if [[ "$mu_type" == "static" ]]; then
            sed -i "62s/.*/    one_class: $mu1/" "$config_copy"
            sed -i "63s/.*/    first_last_class: $mu2/" "$config_copy"
            sed -i "64s/.*/    column_one: $mu3/" "$config_copy"
            sed -i "65s/.*/    change_class: $mu4/" "$config_copy"
            sed -i "70s/.*/    monotonicity: $mu5/" "$config_copy"
            sed -i "71s/.*/    concentration: $mu6/" "$config_copy"
            sed -i "72s/.*/    min_thr: $mu7/" "$config_copy"
            sed -i "73s/.*/    max_thr: $mu7/" "$config_copy"           
        fi

        # run script
        echo "n: $n, m: $m, run $((run+1)), default=${default[run]}"
        python cost_function.py "$config_copy" > "output/annealing/${output_folder}/${file_tag}_05_${n}_${m}_${sweeps}_run${run}.txt" 2>&1 &

    done
    wait

done < "$param_file"

rm -f output/annealing/${output_folder}/config_*_run*.yaml
