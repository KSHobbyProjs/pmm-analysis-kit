#!/usr/bin/bash


SEEDS=(472 815 129 604 38 957 281 743 516 199 10 5 0)

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[i]}"
    echo "working through seed $seed"
    ./run_pmm.py sample.h5 --config-file config.txt -L='-2.0,2.0:50' -c dim=2,seed="$seed" -o "pmm_predicted$i.dat" -q --no-normalize
done

