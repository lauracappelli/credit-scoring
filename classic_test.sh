#!/bin/bash

file="config.yaml"

counterparts=(48 52)
grades=(6 6)

for i in "${!counterparts[@]}"; do
    n=${counterparts[i]}
    m=${grades[i]}

    sed -i "19s/.*/n_counterpart: $n/" "$file"
    sed -i "20s/.*/grades: $m/" "$file"

    for run in {1..3}; do
        python classical_approach.py config.yaml > "output/output_${n}_${m}_run${run}.txt" 2>&1 &
    done
    wait
done

