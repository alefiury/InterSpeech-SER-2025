#!/bin/bash
GPU_ID=3
# CONFIG_PATH=("../config/default_last_layer_embedding_finetuning_wavlm.yaml" \
#             "../config/default_last_layer_embedding_finetuning_wav2vec2_300m.yaml" \
#             "../config/default_last_layer_embedding_finetuning_wav2vec2_1b.yaml" \
#             "../config/default_last_layer_embedding_finetuning_hubert_large.yaml" \
#             "../config/default_last_layer_embedding_finetuning_hubert_xlarge.yaml")

CONFIG_PATH=("../config/default_last_layer_embedding_finetuning_wavlm.yaml" \
            "../config/default_last_layer_embedding_finetuning_wav2vec2_300m.yaml" \
            "../config/default_last_layer_embedding_finetuning_hubert_large.yaml")

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python main.py -c=$i -g=$GPU_ID
done
