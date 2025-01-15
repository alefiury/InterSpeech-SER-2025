#!/bin/bash

# Define variables based on the arguments
# TEST_METADATA_PATH="/raid/alefiury/SER/InterSpeech2025/InterSpeech-SER-2025/Dataset/complete_validation.csv"
TEST_METADATA_PATH="/raid/alefiury/SER/InterSpeech2025/InterSpeech-SER-2025/Dataset/complete_validation.csv"
FILENAME_COLUMN="FileName"
TARGET_COLUMN="EmoClass"
# BASE_DIR="/raid/alefiury/SER/InterSpeech2025/challenge_dataset/Audio_XEUS"
BASE_DIR="/raid/alefiury/SER/InterSpeech2025/challenge_dataset/wavlm-large"

# CONFIG_PATH="/raid/alefiury/SER/InterSpeech2025/InterSpeech-SER-2025/config/default_last_layer_embedding_finetuning_xeus.yaml"
CONFIG_PATH="/raid/alefiury/SER/InterSpeech2025/InterSpeech-SER-2025/config/default_last_layer_embedding_finetuning_wavlm_text.yaml"
CHECKPOINT_PATH="/raid/alefiury/SER/InterSpeech2025/InterSpeech-SER-2025/src/InterSpeech-SER-2025/8xs0h6c5/checkpoints/last.ckpt"

GPU=6

BATCH_SIZE=1
NUM_WORKERS=12

echo "ID: " $CHECKPOINT_PATH

echo "Filtered test set"

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

# TEST_METADATA_PATH="/raid/alefiury/SER/InterSpeech2025/InterSpeech-SER-2025/Dataset/balanced_transcribed_canary_test_set.csv"

# echo "================="
# echo "Balanced test set"

# python eval/inference.py \
#     -t "$TEST_METADATA_PATH" \
#     --filename-column "$FILENAME_COLUMN" \
#     --target-column "$TARGET_COLUMN" \
#     --base-dir "$BASE_DIR" \
#     -c "$CONFIG_PATH" \
#     -g "$GPU" \
#     -ck "$CHECKPOINT_PATH" \
#     --batch-size "$BATCH_SIZE" \
#     --num-workers "$NUM_WORKERS"