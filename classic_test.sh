#!/bin/bash

file="config.yaml"

# counterparts=(8 12 14 17 20 25 32* 40* 48 52)
# grades=(3 3 4 4 5 5 5* 6* 6 6)
counterparts=(32 40)
grades=(5 6)

for i in "${!counterparts[@]}"; do
    n=${counterparts[i]}
    m=${grades[i]}

    sed -i "19s/.*/n_counterpart: $n/" "$file"
    sed -i "20s/.*/grades: $m/" "$file"

    for run in {1..3}; do

        config_copy="output/classic_test/test2/config_${n}_${m}_run${run}.yaml"
        cp "$file" "$config_copy"

        def1=$((n-1))
        def2=$((n-run))
        
        if [ "$n" -gt 39 ]; then
            def3=$((n-6))
            sed -i "26s/\[.*\]/[$def3,$def2,$def1]/" "$config_copy"
        else
            sed -i "26s/\[.*\]/[$def2,$def1]/" "$config_copy"
        fi
        
        echo "Running test n=$n, m=$m, run=$run"
        python classical_approach.py "$config_copy" > "output/classic_test/test2/output_${n}_${m}_run${run}.txt" 2>&1 &
    done
    wait
done

rm -f output/classic_test/test2/config_*_run*.yaml