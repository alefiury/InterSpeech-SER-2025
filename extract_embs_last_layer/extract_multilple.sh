#!/bin/bash
CUDA_DEVICE=6
BASE_DIR="/raid/alefiury/SER/InterSpeech2025/challenge_dataset/"
INPUT_DIR="Audios"

# Wav2vec2
# MODEL_NAME="wav2vec2-xls-r-300m"
# OUTPUT_DIR="xls-r-300m"
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_wav2vec_embeddings.py \
#     -b=$BASE_DIR \
#     -i=$INPUT_DIR \
#     -o=$OUTPUT_DIR \
#     -m=$MODEL_NAME

# MODEL_NAME="wav2vec2-xls-r-1b"
# OUTPUT_DIR="xls-r-1b"
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_wav2vec_embeddings.py \
#     -b=$BASE_DIR \
#     -i=$INPUT_DIR \
#     -o=$OUTPUT_DIR \
#     -m=$MODEL_NAME

# HUBERT
# MODEL_NAME="hubert-large-ll60k"
# OUTPUT_DIR="hubert-large"
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_hubert_embeddings.py \
#     -b=$BASE_DIR \
#     -i=$INPUT_DIR \
#     -o=$OUTPUT_DIR \
#     -m=$MODEL_NAME

# MODEL_NAME="hubert-xlarge-ll60k"
# OUTPUT_DIR="hubert-xlarge"
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_hubert_embeddings.py \
#     -b=$BASE_DIR \
#     -i=$INPUT_DIR \
#     -o=$OUTPUT_DIR \
#     -m=$MODEL_NAME

# TitaNet
# OUTPUT_DIR="TitaNet-large"
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_titanet_embeddings.py \
#     -b=$BASE_DIR \
#     -i=$INPUT_DIR \
#     -o=$OUTPUT_DIR

# # WavLM
# MODEL_NAME="wavlm-large"
# OUTPUT_DIR="wavlm-large"
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_wavlm_embeddings.py \
#     -b=$BASE_DIR \
#     -i=$INPUT_DIR \
#     -o=$OUTPUT_DIR \
#     -m=$MODEL_NAME

# Whisper
MODEL_NAME="whisper-large-v3"
OUTPUT_DIR="whisper-large-v3"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_whisper_embeddings.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    -o=$OUTPUT_DIR \
    -m=$MODEL_NAME
