#!/bin/bash
CUDA_DEVICE=5

CONFIG_PATH=("../config/default_lle_bimodal_ms.yaml" \
            "../config/default_lle_bimodal_ms_2.yaml")

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python main.py -c=$i
done