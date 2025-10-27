#!/bin/bash

param_file="output/classic_test/parameter.csv"
config_file="config.yaml"

while IFS= read -r line; do

    n=$(echo "$line" | awk -F',' '{print $1}')
    m=$(echo "$line" | awk -F',' '{print $2}')

    defs=($(echo "$line" | grep -oP '\[[^\]]*\]'))
    def1=${defs[0]}
    def2=${defs[1]}
    def3=${defs[2]}

    echo "n: $n, m: $m, Default: $def1, $def2, $def3"

    sed -i "19s/.*/n_counterpart: $n/" "$config_file"
    sed -i "20s/.*/grades: $m/" "$config_file"

    default=($def1 $def2 $def3)

    for run in {0..2}; do
        run_id=$run+1
        config_copy="output/classic_test/config_${n}_${m}_run$(($run+1)).yaml"
        cp "$config_file" "$config_copy"
        one_indices="one_indices: ${default[run]}"
        sed -i "26s|.*|$one_indices|" "$config_copy"

        echo "n: $n, m: $m, run $((run+1)), default=${default[run]}"
        python classical_approach.py "$config_copy" > "output/classic_test/output_${n}_${m}_run${run}.txt" 2>&1 &
    done
    wait

done < "$param_file"

rm -f output/classic_test/config_*_run*.yaml