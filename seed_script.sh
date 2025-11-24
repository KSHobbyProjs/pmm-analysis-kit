#!/usr/bin/bash


SEEDS=(472 815 129 604 38 957 281 743 516 199 10 5 0 374 950 731 598 156 155 58 866 601 708)

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[i]}"
    echo "working through seed $seed"
    ./run_pmm.py sample.dat --config-file config.txt -c dim=6,num_primary=3,seed=$seed --epochs 30000 -L 5.0,20.0:150 -o pmm_predictions$seed.dat --save-loss loss$seed.dat -q
done

