#!/bin/bash
DEVICE="cuda"
BASE_DIR="/hadatasets/alef.ferreira/SER/Interspeech"
INPUT_DIR="Audios"
NUM_WORKERS=32

CUDA_DEVICE=4

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_f.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    --num-workers=$NUM_WORKERS \
    --device=$DEVICE
