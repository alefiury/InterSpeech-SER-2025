#!/bin/bash

GPU=6
BATCH_SIZE=1
NUM_WORKERS=12
TEST_METADATA_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/Dataset/transcribed_canary_unlabeled_train.csv"
FILENAME_COLUMN="FileName"
TARGET_COLUMN="EmoClass"
BASE_DIR="/hadatasets/alef.ferreira/SER/Interspeech/Audios"

CONFIG_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/config_runs/main.yaml"
CHECKPOINT_PATH="/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/InterSpeech-SER-2025/ons35ii0/checkpoints/epoch=06-val/f1-score=0.3999.ckpt"

# Run the Python script with the defined arguments
python eval/build_test_submission_runs.py \
    -t "$TEST_METADATA_PATH" \
    --filename-column "$FILENAME_COLUMN" \
    --target-column "$TARGET_COLUMN" \
    --base-dir "$BASE_DIR" \
    -c "$CONFIG_PATH" \
    -g "$GPU" \
    -ck "$CHECKPOINT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"