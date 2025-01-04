#!/bin/bash

# Define variables based on the arguments
TEST_METADATA_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/Dataset/balanced_transcribed_canary_test_set.csv"
FILENAME_COLUMN="FileName"
TARGET_COLUMN="EmoClass"
BASE_DIR="/hadatasets/alef.ferreira/SER/Interspeech/Audios"

CONFIG_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/config/default_finetuning_multimodal_spkemb.yaml"
CHECKPOINT_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/InterSpeech-SER-2025/smxofpnw/checkpoints/last.ckpt"

GPU=6

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