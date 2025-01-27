#!/bin/bash
CUDA_DEVICE=7

CONFIG_PATH=("../config/default_lle_bimodal.yaml" \
            "../config/default_lle_bimodal_2.yaml")

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python main.py -c=$i
done