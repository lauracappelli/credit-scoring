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
        config_copy="output/new_qubo_test/config_${n}_${m}_run${run}.yaml"
        cp "$config_file" "$config_copy"
        
        # compute hyperparameters
        one_indices="one_indices: ${default[run]}"
        def_number=$(echo "${default[run]}" | tr -d '[]' | tr ',' ' ' | wc -w)
        mu_mon=$((def_number * 5))
        mu_conc=$((n / m * 10))
        mu_thr=$((mu_conc / 2))

        # TEST 1 
        # mu_1c=$(((n*m)**2))
        # mu_fst_lst=$(((n*m)*5))
        # mu_subm_1000=$(((n*m)*5))
        # mu_subm_0001=$(((n*m)*5))
        # mu_subm_0110=$(((n*m)*5))
        # mu_restart=$(((n*m)*10))
        # mu_glob_col=$(((n*m)*6))
        # mu_change_class=$(((n*m)*6))

        # # TEST 2 
        # mu_1c=$(((n*m)**2))
        # mu_fst_lst=$(((n*m)*5))
        # mu_subm_1000=$(((n*m)*5))
        # mu_subm_0001=$(((n*m)*5))
        # mu_subm_0110=$(((n*m)*5))
        # mu_restart=$(((n*m)*15))
        # mu_glob_col=$(((n*m)*10))
        # mu_change_class=$(((n*m)*10))

        # # TEST 3
        # mu_1c=$(((n*m)**2))
        # mu_fst_lst=$(((n*m)*5))
        # mu_subm_1000=$((0))
        # mu_subm_0001=$((0))
        # mu_subm_0110=$((0))
        # mu_restart=$((0))
        # mu_glob_col=$(((n*m)*30))
        # mu_change_class=$(((n*m)*30))

        # # TEST 4
        # mu_1c=$(((n*m)**2))
        # mu_fst_lst=$(((n*m)*5))
        # mu_subm_1000=$((0))
        # mu_subm_0001=$((0))
        # mu_subm_0110=$((0))
        # mu_restart=$((0))
        # mu_glob_col=$(((n*m)*15))
        # mu_change_class=$(((n*m)*15))

        # # TEST 5 = TEST 4 MA con dwave

        # # TEST 6
        # mu_1c=$(((n*m)**2))
        # mu_fst_lst=$(((n*m)*5))
        # mu_subm_1000=$((0))
        # mu_subm_0001=$((0))
        # mu_subm_0110=$((0))
        # mu_restart=$((0))
        # mu_glob_col=$(((n*m)*40))
        # mu_change_class=$(((n*m)*40))

        # # TEST 7
        # mu_mon=$((def_number * 20))
        # mu_conc=$((n / m * 5))
        # mu_thr=$((mu_conc))

        # # TEST 8
        # mu_mon=$((def_number * 20))
        # mu_conc=$((n / m * 2))
        # mu_thr=$((mu_conc * 5))
        
        # # TEST 8
        # mu_mon=$((def_number * 15))
        # mu_conc=$((n / m))
        # mu_thr=$((mu_conc * 5))

        # # TEST 9
        # mu_mon=$((def_number * 15))
        # mu_conc=$((1))
        # mu_thr=$(((n/m) * 10))

        # # TEST 9bis
        # mu_mon=$((def_number * 15))
        # mu_conc=$((0))
        # mu_thr=$(((n/m) * 10))

        # Nuovi test
        # mu_mon=$((def_number * 10))
        # mu_conc=$((3 * n / m))
        # mu_thr=$((mu_conc / 2))
        
        # mu_1c=$(((n*m)**2))
        # mu_fst_lst=$(((n*m)*5))
        # mu_glob_col=$(((n*m)*40))
        # mu_change_class=$(((n*m)*40))

        # mu_subm_1000=$((0))
        # mu_subm_0001=$((0))
        # mu_subm_0110=$((0))
        # mu_restart=$((0))

        # # Nuovi test 2
        # mu_mon=$((def_number * 12))
        # mu_conc=$((3 * n / m))
        # mu_thr=$((mu_conc / 2))

        # # Nuovi test 3
        # mu_1c=$(((n*m)**2))
        # mu_fst_lst=$(((n*m)*5))
        # mu_glob_col=$(((n*m)*50))
        # mu_change_class=$(((n*m)*50))

        # mu_mon=$((def_number * 12))
        # mu_conc=$((3 * n / m))
        # mu_thr=$((mu_conc / 2))

        # Nuovi test 4
        mu_1c=$(((n*m*2)**2))
        mu_fst_lst=$(((n*m)*5))
        mu_glob_col=$(((n*m)*75))
        mu_change_class=$(((n*m)*75))

        mu_mon=$((def_number * 12))
        mu_conc=$((3 * n / m))
        mu_thr=$((mu_conc / 2))

        # dwave
        # mu_1c=$(((n*m)/4))
        # mu_fst_lst=$(((n*m)*5))
        # mu_glob_col=$(((n*m)*40))
        # mu_change_class=$(((n*m)*40))
        # mu_mon=$((def_number * 5))
        # mu_conc=$((n / m * 10))
        # mu_thr=$((mu_conc/2))

        # Set in config.yaml the correct parameters
        sed -i "19s/.*/n_counterpart: $n/" "$config_copy"
        sed -i "20s/.*/grades: $m/" "$config_copy"
        sed -i "26s|.*|$one_indices|" "$config_copy"
        sed -i "60s/.*/    one_class: $mu_1c/" "$config_copy"
        sed -i "62s/.*/        first_last_class: $mu_fst_lst/" "$config_copy"
        sed -i "63s/.*/        subm_1000: $mu_subm_1000/" "$config_copy"
        sed -i "64s/.*/        subm_0001: $mu_subm_0001/" "$config_copy"
        sed -i "65s/.*/        subm_0110: $mu_subm_0110/" "$config_copy"
        sed -i "66s/.*/        restart: $mu_restart/" "$config_copy"
        sed -i "67s/.*/        column_one: $mu_glob_col/" "$config_copy"
        sed -i "68s/.*/        change_class: $mu_change_class/" "$config_copy"
        sed -i "69s/.*/    monotonicity: $mu_mon/" "$config_copy"
        sed -i "70s/.*/    concentration: $mu_conc/" "$config_copy"
        sed -i "71s/.*/    min_thr: $mu_thr/" "$config_copy"
        sed -i "72s/.*/    max_thr: $mu_thr/" "$config_copy"
        sed -i "85s/.*/shots: $sweeps/" "$config_copy"

        echo "n: $n, m: $m, run $((run+1)), default=${default[run]}"
        # python cost_function.py "$config_copy" > "output/qubo_test3/test4_${n}_${m}_shots${sweeps}_run${run}.txt" 2>&1 &
        python cost_function.py "$config_copy" > "output/new_qubo_test/test06_${n}_${m}_run${run}.txt" 2>&1 &

    done
    wait

done < "$param_file"

rm -f output/new_qubo_test/config_*_run*.yaml