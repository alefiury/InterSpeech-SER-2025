#!/bin/bash
GPU_ID=6
CONFIG_PATH=("../config/default-dynamic-per_layer-att_pool.yaml" \
            "../config/default-dynamic-per_layer-mean_pool-backbone.yaml" \
            "../config/default-dynamic-per_layer-mean_pool.yaml" \
            "../config/default-dynamic-weighted_sum-att_pool.yaml" \
            "../config/default-dynamic-weighted_sum-mean_pool.yaml")

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python main.py -c=$i -g=$GPU_ID
done
