#!/bin/bash

# Define variables based on the arguments
TEST_METADATA_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/Dataset/true_test.csv"
FILENAME_COLUMN="FileName"
TARGET_COLUMN="EmoClass"
BASE_DIR="/hadatasets/alef.ferreira/SER/Interspeech/Audios"

CONFIG_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/config/default_finetuning_multimodal_spkemb_ms.yaml"
CHECKPOINT_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/InterSpeech-SER-2025/ambjo5o9/checkpoints/last.ckpt"

GPU=5

BATCH_SIZE=32
NUM_WORKERS=12

# Run the Python script with the defined arguments
python eval/build_test_submission.py \
    -t "$TEST_METADATA_PATH" \
    --filename-column "$FILENAME_COLUMN" \
    --target-column "$TARGET_COLUMN" \
    --base-dir "$BASE_DIR" \
    -c "$CONFIG_PATH" \
    -g "$GPU" \
    -ck "$CHECKPOINT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"