#!/bin/bash

# Define variables based on the arguments
TEST_METADATA_PATH="/raid/alefiury/SER/InterSpeech2025/wavlm_ravdess_test.csv"
FILENAME_COLUMN="filename"
TARGET_COLUMN="emotion"
BASE_DIR="/raid/alefiury/SER/InterSpeech2025/Audio_Speech_Actors_01-24_wavlm_last_layer"

CONFIG_PATH="/raid/alefiury/SER/InterSpeech2025/InterSpeech-SER-2025/config/default_last_layer_embedding_finetuning.yaml"
CHECKPOINT_PATH="/raid/alefiury/SER/InterSpeech2025/InterSpeech-SER-2025/src/InterSpeech-SER-2025/8vwo0510/checkpoints/last.ckpt"

GPU=7

BATCH_SIZE=32
NUM_WORKERS=12

# Run the Python script with the defined arguments
python eval/inference.py \
    -t "$TEST_METADATA_PATH" \
    --filename-column "$FILENAME_COLUMN" \
    --target-column "$TARGET_COLUMN" \
    --base-dir "$BASE_DIR" \
    -c "$CONFIG_PATH" \
    -g "$GPU" \
    -ck "$CHECKPOINT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"